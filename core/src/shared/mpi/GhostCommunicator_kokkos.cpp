#include "GhostCommunicator.h"
#include <cstdint>
#include "utils/mpiUtils/GlobalMpiSession.h"

namespace dyablo{
namespace muscl_block{

GhostCommunicator_kokkos::GhostCommunicator_kokkos( std::shared_ptr<AMRmesh> amr_mesh )
{
  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();

  //! Get map that contains ghost to send to each rank from PABLO : rank -> [iOcts]
  const std::map<int, std::vector<uint32_t>>& ghost_map = amr_mesh->getBordersPerProc();

  //Compute send sizes
  {
    Kokkos::realloc(this->send_sizes, nb_proc);
    this->send_sizes_host = Kokkos::create_mirror_view(this->send_sizes);  
    uint32_t nbGhostSend = 0; 
    for(int rank=0; rank<nb_proc; rank++)
    {
      auto it = ghost_map.find(rank);
      if(it==ghost_map.end())
      {
        send_sizes_host(rank) = 0;
      }
      else
      {
        const std::vector<uint32_t>& iOcts_source = it->second;
        send_sizes_host(rank) = iOcts_source.size();
        nbGhostSend += iOcts_source.size();
      }
    }
    // Copy number of octants to recieve to device (host + device are up to date)
    Kokkos::deep_copy(send_sizes, send_sizes_host);

    // Allocate buffer to contain iOcts to send
    Kokkos::realloc(this->send_iOcts, nbGhostSend);
  }

  //Fill list of ghost octants to send
  {
    Kokkos::View<uint32_t*>::HostMirror send_iOcts_host = Kokkos::create_mirror_view(this->send_iOcts);
    uint32_t iOct_offset = 0;
    for( auto& p : ghost_map )
    {
      const std::vector<uint32_t>& iOcts_source = p.second;
      for( size_t i=0; i<iOcts_source.size(); i++ )
      {
        send_iOcts_host(iOct_offset + i) = iOcts_source[i];
      }
      iOct_offset += iOcts_source.size();
    }
    // Move send_iOcts to device
    Kokkos::deep_copy(this->send_iOcts, send_iOcts_host);
  }

  // Fill number of octants to recieve
  {
    Kokkos::realloc(this->recv_sizes, nb_proc);
    this->recv_sizes_host = Kokkos::create_mirror_view(this->recv_sizes);
    MPI_Alltoall( send_sizes_host.data(), 1, MPI_INT, 
                  recv_sizes_host.data(), 1, MPI_INT,
                  MPI_COMM_WORLD );
    // Copy number of octants to recieve to device (host + device are up to date)
    Kokkos::deep_copy(recv_sizes, recv_sizes_host);
    this->nbghosts_recv = 0;
    for(int i=0; i<nb_proc; i++)
      this->nbghosts_recv += recv_sizes_host(i);
  }
}

namespace{
  template<typename DataArray_t>
  std::enable_if_t< DataArray_t::rank == 1, void> 
  copy_element(  const DataArray_t& Udest, uint32_t iOct_dest, 
                      const DataArray_t& Usrc , uint32_t iOct_src,
                      uint32_t elt_index)
  {
    Udest(iOct_dest) = Usrc(iOct_src);
  }

  template<typename DataArray_t>
  std::enable_if_t< DataArray_t::rank == 2, void> 
  copy_element(  const DataArray_t& Udest, uint32_t iOct_dest, 
                      const DataArray_t& Usrc , uint32_t iOct_src,
                      uint32_t elt_index)
  {
    Udest(elt_index, iOct_dest) = Usrc(elt_index, iOct_src);
  }

  template<typename DataArray_t>
  std::enable_if_t< DataArray_t::rank == 3, void> 
  copy_element( const DataArray_t& Udest, uint32_t iOct_dest, 
                const DataArray_t& Usrc , uint32_t iOct_src,
                uint32_t elt_index)
  {
    uint32_t i1 = elt_index / Udest.extent(0);
    uint32_t i0 = elt_index % Udest.extent(0);

    Udest(i0, i1, iOct_dest) = Usrc(i0, i1, iOct_src);
  }

  template<typename DataArray_t>
  std::enable_if_t< DataArray_t::rank == 1, DataArray_t> 
  get_subview( const DataArray_t& U, uint32_t iOct_begin, uint32_t iOct_end)
  {
    return Kokkos::subview( U, std::make_pair(iOct_begin, iOct_end) );
  }

  template<typename DataArray_t>
  std::enable_if_t< DataArray_t::rank == 2, DataArray_t> 
  get_subview( const DataArray_t& U, uint32_t iOct_begin, uint32_t iOct_end)
  {
    return Kokkos::subview( U, Kokkos::ALL, 
                            std::make_pair(iOct_begin,iOct_end) );
  }

  template<typename DataArray_t>
  std::enable_if_t< DataArray_t::rank == 3, DataArray_t> 
  get_subview( const DataArray_t& U, uint32_t iOct_begin, uint32_t iOct_end)
  {
    return Kokkos::subview( U, Kokkos::ALL, Kokkos::ALL, 
                            std::make_pair(iOct_begin,iOct_end) );
  }

  template<typename T> MPI_Datatype get_MPI_Datatype()
  {
    static_assert(!std::is_same<T,T>::value, "Please add type to get_MPI_Datatype");
    return 0;
  }

  template<> MPI_Datatype get_MPI_Datatype<double>() {return MPI_DOUBLE;}
  //template<> MPI_Datatype get_MPI_Datatype<float>() {return MPI_FLOAT;}
  template<> MPI_Datatype get_MPI_Datatype<unsigned short>() {return MPI_UNSIGNED_SHORT;}
  template<> MPI_Datatype get_MPI_Datatype<int>() {return MPI_INT;}
} // namespace

template< typename DataArray_t >
void GhostCommunicator_kokkos::exchange_ghosts_aux( const DataArray_t& U, DataArray_t& Ughost) const
{ 
  static_assert( std::is_same<typename DataArray_t::array_layout, Kokkos::LayoutLeft>::value, 
                 "This implementation of GhostCommunicator_kokkos only supports Kokkos::LayoutLeft views" );

  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
  uint32_t nbGhostsSend = send_iOcts.size();
  

  constexpr int dim = DataArray_t::rank;
  // Allocate send_buffers with same dimension for each octant and enough octants for send buffer
  Kokkos::LayoutLeft extents_send = U.layout();
  extents_send.dimension[dim-1] = nbGhostsSend; 
  DataArray_t send_buffers("Send buffers", extents_send); 

  uint32_t elts_per_octs = 1;
  for(int i=0; i<dim-1; i++)
     elts_per_octs *= U.extent(i);

  const Kokkos::View<uint32_t*> & send_iOcts_ref = send_iOcts;
  Kokkos::parallel_for( "GhostCommunicator::fill_send_buffer", send_buffers.size(),
                        KOKKOS_LAMBDA(uint32_t index)
  {
    uint32_t iGhost = index/elts_per_octs;
    uint32_t iOct_origin = send_iOcts_ref(iGhost);
    uint32_t i = index%elts_per_octs;

    copy_element( send_buffers, iGhost, U, iOct_origin, i );
  });

  // Resize Ughost with same dimension for each octant and enough octants for ghosts
  {
  Kokkos::LayoutLeft extents_recv = U.layout();
  extents_recv.dimension[dim-1] = this->nbghosts_recv; 
  Kokkos::realloc(Ughost, extents_recv);
  }
  //#define MPI_IS_CUDA_AWARE
  #ifdef MPI_IS_CUDA_AWARE
  using MPIBuffer_t = DataArray_t;
  Kokkos::fence();
  MPIBuffer_t& mpi_send_buffers = send_buffers;
  // Ughost in PABLO is filled in source rank order : no need for another copy
  MPIBuffer_t& mpi_recv_buffers = Ughost;
  #else
  using MPIBuffer_t = typename DataArray_t::HostMirror;
  MPIBuffer_t mpi_send_buffers = Kokkos::create_mirror_view(send_buffers);
  Kokkos::deep_copy(mpi_send_buffers, send_buffers);
  MPIBuffer_t mpi_recv_buffers = Kokkos::create_mirror_view(Ughost);
  #endif

  MPI_Datatype mpi_type = get_MPI_Datatype<typename DataArray_t::value_type>();
  std::vector<MPI_Request> mpi_requests;
  // Post MPI_Isends
  {
    uint32_t iOct_offset = 0;
    for(int rank=0; rank<nb_proc; rank++)
    {
      uint32_t iOct_range_end = iOct_offset+send_sizes_host(rank);
      // This only works if DataArray_t is LayoutLeft
      // Otherwise, subview is not contiguous and this is necessary for MPI
      MPIBuffer_t send_buffer_rank 
        = get_subview( mpi_send_buffers, iOct_offset, iOct_range_end);

      mpi_requests.push_back(MPI_REQUEST_NULL);
      MPI_Isend( send_buffer_rank.data(), send_buffer_rank.size(), mpi_type,
                rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );

      iOct_offset += send_sizes_host(rank);
    }
  }

  // Post MPI_Irecv to store recieved ghosts directly in Ughost
  {
    uint32_t iOct_offset = 0;
    
    for(int rank=0; rank<nb_proc; rank++)
    {
      uint32_t iOct_range_end = iOct_offset+recv_sizes_host(rank);
      // This only works if DataArray_t is LayoutLeft
      // Otherwise, subview is not contiguous and this is necessary for MPI
      MPIBuffer_t recv_buffer_rank 
        = get_subview( mpi_recv_buffers, iOct_offset, iOct_range_end );
      mpi_requests.push_back(MPI_REQUEST_NULL);
      MPI_Irecv( recv_buffer_rank.data(), recv_buffer_rank.size(), mpi_type,
                rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );

      iOct_offset += recv_sizes_host(rank);
    }
  }

  MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);
  #ifdef MPI_IS_CUDA_AWARE
  Kokkos::fence();
  #else
  Kokkos::deep_copy(Ughost, mpi_recv_buffers);
  #endif
}

void GhostCommunicator_kokkos::exchange_ghosts(const DataArrayBlock& U, DataArrayBlock& Ughost) const
{
  exchange_ghosts_aux(U, Ughost);
}

void GhostCommunicator_kokkos::exchange_ghosts(const Kokkos::View<uint16_t**, Kokkos::LayoutLeft>& U, Kokkos::View<uint16_t**, Kokkos::LayoutLeft>& Ughost) const
{
  exchange_ghosts_aux(U, Ughost);
}

void GhostCommunicator_kokkos::exchange_ghosts(const Kokkos::View<int*, Kokkos::LayoutLeft>& U, Kokkos::View<int*, Kokkos::LayoutLeft>& Ughost) const
{
  exchange_ghosts_aux(U, Ughost);
}

}//namespace muscl_block
}//namespace dyablo
