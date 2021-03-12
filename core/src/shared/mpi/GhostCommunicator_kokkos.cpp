#include "GhostCommunicator.h"
#include <cstdint>
#include "utils/mpiUtils/GlobalMpiSession.h"

namespace dyablo{
namespace muscl_block{

GhostCommunicator_kokkos::GhostCommunicator_kokkos( const std::map<int, std::vector<uint32_t>>& ghost_map )
{
  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();

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
  template<int N, typename DataArray_t, typename... Args>
  KOKKOS_INLINE_FUNCTION
  std::enable_if_t< DataArray_t::rank-1 == N , void> 
  copy_element_aux( const DataArray_t& Udest, uint32_t iOct_dest, 
                    const DataArray_t& Usrc , uint32_t iOct_src,
                    uint32_t elt_index, Args... is)
  {
    Udest(is..., iOct_dest) = Usrc(is..., iOct_src);
  }
  template<int N, typename DataArray_t, typename... Args>
  KOKKOS_INLINE_FUNCTION
  std::enable_if_t< DataArray_t::rank-1 != N , void> 
  copy_element_aux( const DataArray_t& Udest, uint32_t iOct_dest, 
                    const DataArray_t& Usrc , uint32_t iOct_src,
                    uint32_t elt_index, Args... is)
  {
    uint32_t current_dim_size = Udest.extent(N);
    uint32_t rem = elt_index%current_dim_size;
    uint32_t div = elt_index/current_dim_size;
    copy_element_aux<N+1>(Udest, iOct_dest, Usrc, iOct_src, div, is..., rem);
  }
  /**
   * Generic way copy octant data in an n-dimensional Kokkos view 
   * form Usrc to Udest
   * @param elt_inex is the linearized index to elements of octant
   **/
  template<typename DataArray_t>
  KOKKOS_INLINE_FUNCTION
  void copy_element(  const DataArray_t& Udest, uint32_t iOct_dest, 
                      const DataArray_t& Usrc , uint32_t iOct_src,
                      uint32_t elt_index)
  {
    copy_element_aux<0>(Udest, iOct_dest, Usrc, iOct_src, elt_index);
  }


  /**
   * Generic way to get a subview of an an n-dimensional Kokkos view
   * with Kokkos::ALL, ..., std::make_pair(iOct_begin, iOct_end) as parameters
   **/
  template<typename DataArray_t, typename... Args>
  std::enable_if_t< sizeof...(Args) == DataArray_t::rank-1, DataArray_t > 
  get_subview( const DataArray_t& U, uint32_t iOct_begin, uint32_t iOct_end, Args... is)
  {
    return Kokkos::subview( U, is..., std::make_pair(iOct_begin, iOct_end) );
  }
  template<typename DataArray_t, typename... Args>
  std::enable_if_t< sizeof...(Args) != DataArray_t::rank-1, DataArray_t > 
  get_subview( const DataArray_t& U, uint32_t iOct_begin, uint32_t iOct_end, Args... is)
  {
    return get_subview(U, iOct_begin, iOct_end, Kokkos::ALL(), is...);
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

  /**
   * Slice view into one subview per rank.
   * Each subview `res[rank]` contains `sizes[rank]` octants from `view`
   * 
   * This is the case where MPIBuffer and DataArray_t are the same type : give direct access to `view`
   * This is used on CPU-only or when MPI is CUDA-Aware
   **/
  template <typename DataArray_t>
  std::vector<DataArray_t> get_subviews(const DataArray_t& view, const Kokkos::View<uint32_t*>::HostMirror& sizes)
  {
    int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
    
    std::vector<DataArray_t> res(nb_proc);
    uint32_t iOct_offset = 0;
    for(int rank=0; rank<nb_proc; rank++)
    {
      uint32_t iOct_range_end = iOct_offset+sizes(rank);
      // This only works if DataArray_t is LayoutLeft
      // Otherwise, subview is not contiguous and this is necessary for MPI
      res[rank] = get_subview( view, iOct_offset, iOct_range_end);

      iOct_offset += sizes(rank);
    }

    Kokkos::fence();

    return res;
  }

  /**
   * Slice view into one subview per rank.
   * Each subview `res[rank]` contains `sizes[rank]` octants from `view`
   * 
   * This is the case where MPIBuffer is DataArray_t::HostMirror : data from `view` has to be deep_copied
   * This is used with Kokkos::CUDA when MPI is not CUDA-Aware
   **/
  template <typename MPIBuffer_t, typename DataArray_t>
  std::enable_if_t< std::is_same< MPIBuffer_t, typename DataArray_t::HostMirror>::value,  
  std::vector<MPIBuffer_t>> get_subviews(const DataArray_t& view, const Kokkos::View<uint32_t*>::HostMirror& sizes)
  {
    int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
    
    MPIBuffer_t view_host = Kokkos::create_mirror_view(view);
    Kokkos::deep_copy(view_host, view);

    std::vector<MPIBuffer_t> res(nb_proc);
    uint32_t iOct_offset = 0;
    for(int rank=0; rank<nb_proc; rank++)
    {
      uint32_t iOct_range_end = iOct_offset+sizes(rank);
      // This only works if DataArray_t is LayoutLeft
      // Otherwise, subview is not contiguous and this is necessary for MPI
      res[rank] = get_subview( view_host, iOct_offset, iOct_range_end);

      iOct_offset += sizes(rank);
    }

    return res;
  }


  /**
   * Copy Octants `send_iOcts` from U to the MPI buffers
   * @returns the list of buffers ready to be sent to every rank
   **/
  template <typename MPIBuffer_t, typename DataArray_t>
  std::vector<MPIBuffer_t> pack( const DataArray_t& U, const Kokkos::View<uint32_t*>& send_iOcts, const Kokkos::View<uint32_t*>::HostMirror& send_sizes_host )
  {
    // When iOct is the rightmost subscript in U, a unique view of size (:,...,sum(send_sizes_host)) is 
    // allocated and then sliced in subviews for each rank

    static_assert( std::is_same<typename DataArray_t::array_layout, Kokkos::LayoutLeft>::value, 
                 "This implementation of GhostCommunicator_kokkos only supports Kokkos::LayoutLeft views" ); 

    constexpr int dim = DataArray_t::rank;
    int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
    
    // Allocate send_buffers with same dimension for each octant but with sum(send_sizes_host) octants
    Kokkos::LayoutLeft extents_send = U.layout();
    extents_send.dimension[dim-1] = send_iOcts.size(); 
    DataArray_t send_buffers("Send buffers", extents_send); 

    uint32_t elts_per_octs = 1;
    for(int i=0; i<dim-1; i++)
      elts_per_octs *= U.extent(i);

    // Copy values to send from U to send_buffers
    Kokkos::parallel_for( "GhostCommunicator::fill_send_buffer", send_buffers.size(),
                          KOKKOS_LAMBDA(uint32_t index)
    {
      uint32_t iGhost = index/elts_per_octs;
      uint32_t iOct_origin = send_iOcts(iGhost);
      uint32_t i = index%elts_per_octs;

      copy_element( send_buffers, iGhost, U, iOct_origin, i );
    });

    return get_subviews<MPIBuffer_t>(send_buffers, send_sizes_host);    
  }


  /**
   * Transfert values from recv_buffers to Ughost
   * 
   * This is the case where MPIBuffer and DataArray_t are the same type : recv_buffers has direct access to Ughost
   * This is used on CPU-only or when MPI is CUDA-Aware
   **/
  template <typename DataArray_t>
  void unpack( const std::vector<DataArray_t>& recv_buffers, const DataArray_t& Ughost, const Kokkos::View<uint32_t*>::HostMirror& recv_sizes_host )
  {
    // When iOct is the rightmost subscript in Ughost, Ughost is directly used as recieve buffer
    // There is no copy to perform

    Kokkos::fence();
  }

  /**
   * Transfert values from recv_buffers to Ughost
   * 
   * This is the case where MPIBuffer is DataArray_t::HostMirror : data from `recv_buffers` has to be deep_copied to Ughost
   * This is used with Kokkos::CUDA when MPI is not CUDA-Aware
   **/
  template <typename MPIBuffer_t, typename DataArray_t>
  std::enable_if_t< std::is_same< MPIBuffer_t, typename DataArray_t::HostMirror>::value, void >
  unpack( const std::vector<MPIBuffer_t>& recv_buffers, const DataArray_t& Ughost, const Kokkos::View<uint32_t*>::HostMirror& recv_sizes_host )
  {
    // When iOct is the rightmost subscript in Ughost, Ughost is directly used as recieve buffer
    // When MPIBuffer is on host (not CUDA-Aware) we need to copy back to device

    std::vector<DataArray_t> recv_buffers_device = get_subviews<DataArray_t>(Ughost, recv_sizes_host);

    int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
    for(int i=0; i<nb_proc; i++)
    {
      Kokkos::deep_copy(recv_buffers_device[i], recv_buffers[i]);
    }
  }

} // namespace

template< typename DataArray_t >
void GhostCommunicator_kokkos::exchange_ghosts_aux( const DataArray_t& U, DataArray_t& Ughost) const
{ 
#ifdef MPI_IS_CUDA_AWARE    
  using MPIBuffer = DataArray_t;
#else
  using MPIBuffer = typename DataArray_t::HostMirror;
#endif

  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();

  // Pack send buffers from U, allocate recieve buffers
  std::vector<MPIBuffer> send_buffers = pack<MPIBuffer>( U, this->send_iOcts, this->send_sizes_host );

  Kokkos::LayoutLeft extents_recv = U.layout();
  extents_recv.dimension[DataArray_t::rank-1] = this->nbghosts_recv; 
  Kokkos::realloc(Ughost, extents_recv);

  std::vector<MPIBuffer> recv_buffers = get_subviews<MPIBuffer>(Ughost, recv_sizes_host);
  
  {
    MPI_Datatype mpi_type = get_MPI_Datatype<typename DataArray_t::value_type>();
    std::vector<MPI_Request> mpi_requests;
    // Post MPI_Isends
    for(int rank=0; rank<nb_proc; rank++)
    {
      if( send_buffers[rank].size() > 0 )
      {
        mpi_requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend( send_buffers[rank].data(), send_buffers[rank].size(), mpi_type,
                  rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );
      }
    }
    // Post MPI_Irecv   
    for(int rank=0; rank<nb_proc; rank++)
    {
      if( recv_buffers[rank].size() > 0 )
      {
        mpi_requests.push_back(MPI_REQUEST_NULL);
        MPI_Irecv( recv_buffers[rank].data(), recv_buffers[rank].size(), mpi_type,
                  rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );
      }
    }
    MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);
  }

  // Unpack recv buffers to Ughost
  unpack(recv_buffers, Ughost, this->recv_sizes_host);
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

void GhostCommunicator_kokkos::exchange_ghosts(const DataArray& U, DataArray& Ughost) const
{
  exchange_ghosts_aux(U, Ughost);
}

}//namespace muscl_block
}//namespace dyablo
