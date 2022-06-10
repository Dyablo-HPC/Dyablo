#pragma once

#include "Kokkos_Core.hpp"
#include "GhostCommunicator.h"

namespace dyablo {

/**
 * Ghost communicator that extracts communication metadata from PABLO 
 * then serialize/deserialize in Kokkos kernels and use CUDA-aware MPI 
 **/
class GhostCommunicator_kokkos : public GhostCommunicator_base
{
public:
    GhostCommunicator_kokkos( const Kokkos::View< int* > domains, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() );
    GhostCommunicator_kokkos( const std::map<int, std::vector<uint32_t>>& ghost_map, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() );
    GhostCommunicator_kokkos( std::shared_ptr<AMRmesh> amr_mesh, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
     : GhostCommunicator_kokkos(amr_mesh->getBordersPerProc(), mpi_comm)
    {}

    GhostCommunicator_kokkos( std::shared_ptr<AMRmesh_hashmap> amr_mesh, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
     : GhostCommunicator_kokkos(amr_mesh->getBordersPerProc(), mpi_comm)
    {}
    
    /// @copydoc GhostCommunicator_base::getNumGhosts
    uint32_t getNumGhosts() const;

    /**
     * Generic function to exchange octant data stored un a Kokkos View
     * @tparam is the Kokkos::View type. It must be Kokkos::LayoutLeft
     * @tparam iOct_pos position of iOct coordinate in U. When iOct is not leftmost 
     *         coordinate (iOct_pos = DataArray_t::rank-1), packing/unpacking is less efficient 
     *         because data needs to be transposed.
     * @param U is local octant data with iOct_pos-nth subscript the octant index
     * @param Ughost is the ghost octant data to fill, it will be resized to match the number of ghost octants
     **/
    template< int iOct_pos, typename DataArray_t >
    void exchange_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const;

private:
    Kokkos::View<uint32_t*> recv_sizes, send_sizes; //!Number of octants to send/recv for each proc
    Kokkos::View<uint32_t*>::HostMirror recv_sizes_host, send_sizes_host; //!Number of octants to send/recv for each proc
    Kokkos::View<uint32_t*> send_iOcts; //! List of octants to send (first send_sizes[0] iOcts to send to rank[0] and so on...)
    uint32_t nbghosts_recv;
    MpiComm mpi_comm;    
};

namespace GhostCommunicator_kokkos_impl{
  using namespace userdata_utils;  

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
    int nb_proc = sizes.size();
    
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
    int nb_proc = sizes.size();
    
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
   * @tparam iOct_pos position of coorinate iOct in U
   * @returns the list of buffers ready to be sent to every rank, buffers are transposed from U to have iOct as leftmost subscript
   **/
  template <typename MPIBuffer_t, int iOct_pos, typename DataArray_t>
  std::vector<MPIBuffer_t> pack( const DataArray_t& U, const Kokkos::View<uint32_t*>& send_iOcts, const Kokkos::View<uint32_t*>::HostMirror& send_sizes_host )
  {
    // When iOct is the rightmost subscript in U, a unique view of size (:,...,sum(send_sizes_host)) is 
    // allocated and then sliced in subviews for each rank,
    // When iOct is not the rightmost subscript in U, a temporary transposed array is created

    static_assert( std::is_same<typename DataArray_t::array_layout, Kokkos::LayoutLeft>::value, 
                 "This implementation of GhostCommunicator_kokkos only supports Kokkos::LayoutLeft views" ); 

    constexpr int dim = DataArray_t::rank;
    
    // Allocate send_buffers with same dimension for each octant but with sum(send_sizes_host) octants
    // iOct is also displaced to the rightmost coordinate (if it's not already the case)
    Kokkos::LayoutLeft extents_send = U.layout();
    for(int i=iOct_pos; i<dim-1; i++)
      extents_send.dimension[i] = extents_send.dimension[i+1];
    extents_send.dimension[dim-1] = send_iOcts.size(); 
    DataArray_t send_buffers("Send buffers", extents_send); 

    uint32_t elts_per_octs = 1;
    for(int i=0; i<dim; i++)
      if( i!= iOct_pos )
        elts_per_octs *= U.extent(i);

    // Copy values to send from U to send_buffers
    Kokkos::parallel_for( "GhostCommunicator::fill_send_buffer", send_buffers.size(),
                          KOKKOS_LAMBDA(uint32_t index)
    {
      uint32_t iGhost = index/elts_per_octs;
      uint32_t iOct_origin = send_iOcts(iGhost);
      uint32_t i = index%elts_per_octs;
      
      // copy octant data with iOct dimension moved from iOct_pos to DataArray_t::rank-1
      get_U<DataArray_t::rank-1>(send_buffers, iGhost, i) = get_U<iOct_pos>(U, iOct_origin, i);
    });

    // Slice send_buffers into subviews
    return get_subviews<MPIBuffer_t>(send_buffers, send_sizes_host);    
  }


  /**
   * Transfert values from recv_buffers to Ughost
   * 
   * This is the case where MPIBuffer and DataArray_t are the same type : recv_buffers has direct access to Ughost
   * This is used on CPU-only or when MPI is CUDA-Aware
   * (When iOct is rightmost index)
   **/
  template <typename DataArray_t>
  void unpack( const std::vector<DataArray_t>& /*recv_buffers*/, const DataArray_t& /*Ughost*/, const Kokkos::View<uint32_t*>::HostMirror& /*recv_sizes_host*/ )
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
   * (When iOct is rightmost index)
   **/
  template <typename MPIBuffer_t, typename DataArray_t>
  std::enable_if_t< std::is_same< MPIBuffer_t, typename DataArray_t::HostMirror>::value , 
  void > unpack( const std::vector<MPIBuffer_t>& recv_buffers, const DataArray_t& Ughost, const Kokkos::View<uint32_t*>::HostMirror& recv_sizes_host )
  {
    // When iOct is the rightmost subscript in Ughost, Ughost is directly used as recieve buffer
    // When MPIBuffer is on host (not CUDA-Aware) we need to copy back to device

    std::vector<DataArray_t> recv_buffers_device = get_subviews<DataArray_t>(Ughost, recv_sizes_host);

    int nb_proc = recv_sizes_host.size();
    
    for(int i=0; i<nb_proc; i++)
    {
      Kokkos::deep_copy(recv_buffers_device[i], recv_buffers[i]);
    }
  }

  /**
   * Transfert values from Ughost_right_iOct to Ughost
   * When iOct is not rightmost index in Ughost
   **/
  template <int iOct_pos, typename DataArray_t>
  std::enable_if_t< iOct_pos < DataArray_t::rank-1 , 
  void > transpose( const DataArray_t& Ughost_right_iOct, const DataArray_t& Ughost )
  {
    // When iOct is not the rightmost index, a temportary MPI buffer 
    // with iOct rightmost index is used and has to be transposed

    // Verify Ghost allocation has the right size
    assert( Ughost.extent(iOct_pos) == Ughost_right_iOct.extent(DataArray_t::rank-1) );

    uint32_t elts_per_octs = octant_size<DataArray_t, iOct_pos>(Ughost);

    // Transpose value from Ughost_right_iOct to Ughost
    Kokkos::parallel_for( "GhostCommunicator::unpack_transpose", Ughost_right_iOct.size(),
                          KOKKOS_LAMBDA(uint32_t index)
    {
      uint32_t iOct = index/elts_per_octs;
      uint32_t i = index%elts_per_octs;
      
      get_U<iOct_pos>(Ughost, iOct, i) = get_U<DataArray_t::rank-1>(Ughost_right_iOct, iOct, i);
    });
  }

  /**
   * Transfert values from Ughost_right_iOct to Ughost
   * When iOct is rightmost index un Ughost there is nothing to transpose 
   **/
  template <int iOct_pos, typename DataArray_t>
  std::enable_if_t< iOct_pos == DataArray_t::rank-1 , 
  void > transpose( const DataArray_t& Ughost_right_iOct, const DataArray_t& Ughost )
  {
    Kokkos::deep_copy( Ughost, Ughost_right_iOct );
  }

} // namespace GhostCommunicator_kokkos_impl

template< int iOct_pos, typename DataArray_t >
void GhostCommunicator_kokkos::exchange_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const
{ 
  using namespace GhostCommunicator_kokkos_impl;
  using MPI_Request_t = MpiComm::MPI_Request_t;
#ifdef MPI_IS_CUDA_AWARE    
  using MPIBuffer = DataArray_t;
#else
  using MPIBuffer = typename DataArray_t::HostMirror;
#endif

  int nb_proc = mpi_comm.MPI_Comm_size();

  // Pack send buffers from U, allocate recieve buffers
  std::vector<MPIBuffer> send_buffers = pack<MPIBuffer, iOct_pos>( U, this->send_iOcts, this->send_sizes_host );

  // Allocate Ughost_tmp with same volume of data, but with iOct at rightmost position
  Kokkos::LayoutLeft extents_Ughost_tmp = U.layout();
  for(int i=iOct_pos; i<DataArray_t::rank-1; i++)
    extents_Ughost_tmp.dimension[i] = extents_Ughost_tmp.dimension[i+1];
  extents_Ughost_tmp.dimension[DataArray_t::rank-1] = this->nbghosts_recv;
  DataArray_t Ughost_tmp(U.label()+"_ghost", extents_Ughost_tmp);

  std::vector<MPIBuffer> recv_buffers = get_subviews<MPIBuffer>(Ughost_tmp, recv_sizes_host);
  
  {
    std::vector<MPI_Request_t> mpi_requests;
    // Post MPI_Isends
    for(int rank=0; rank<nb_proc; rank++)
    {
      if( send_buffers[rank].size() > 0 )
      {
        MPI_Request_t r = mpi_comm.MPI_Isend( send_buffers[rank], rank, 0 );
        mpi_requests.push_back(r);
      }
    }
    // Post MPI_Irecv   
    for(int rank=0; rank<nb_proc; rank++)
    {
      if( recv_buffers[rank].size() > 0 )
      {
        MPI_Request_t r = mpi_comm.MPI_Irecv( recv_buffers[rank], rank, 0 );
        mpi_requests.push_back(r);
      }
    }
    mpi_comm.MPI_Waitall(mpi_requests.size(), mpi_requests.data());
  }

  // Unpack recv buffers to Ughost_tmp
  unpack(recv_buffers, Ughost_tmp, this->recv_sizes_host);

  transpose<iOct_pos>( Ughost_tmp, Ughost );
}

} // namespace dyablo