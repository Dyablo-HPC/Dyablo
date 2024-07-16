#pragma once

#include "Kokkos_Core.hpp"
#include "userdata_utils.h"
#include "amr/AMRmesh.h"

namespace dyablo {

/**
 * Communicator to exchange Kokkos::View data between MPI domains. ViewCommunicator can exchange 
 * multidimensionnal views that are distributed in a single dimension. (e.g. cell block arrays 
 * are distributed along the `iOct` dimension, particle arrays are distributed along the iPart dimension).
 * 
 * The distributed dimension is the position of the index in Kokkos::operator() for the distributed objects. 
 * Any index can be the distributed dimension, objects will be automatically packed and unpacked if they 
 * are not contiguous (be aware of performance implications).
 * 
 * ViewCommunicator is constructed with a description of the exchange 
 * pattern (e.g. which local octant to send to which rank), then 
 * exchange_ghosts<distributed_dim>() and reduce_ghosts<distributed_dim>() 
 * are used to perform MPI operations. 
 **/
class ViewCommunicator
{
public: 
    static ViewCommunicator from_mesh( const AMRmesh& mesh, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    {
      return from_mesh(mesh.getMesh(), mpi_comm);
    }

    static ViewCommunicator from_mesh( const AMRmesh_hashmap_new& mesh, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    {
      auto gm = mesh.getGhostMap();
      return ViewCommunicator( gm.send_sizes, gm.send_iOcts, mpi_comm );
    }
    
    static ViewCommunicator from_mesh( const AMRmesh_hashmap& mesh, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    {
      return ViewCommunicator( mesh.getBordersPerProc(), mpi_comm );
    }

    /**
     * Create a new ViewCommunicator using a view containing the target domain for each local object
     * This is for example used for load balancing when blocks are moved from one rank to another
     * (since there is only one target domain per block, this cannot be used for ghosts) 
     * 
     * @param target_domains View containing the target domain for each local object in the distributed 
     *                      dimension (same size as the distributed dimension extent)
     **/
    ViewCommunicator( const Kokkos::View< int* > target_domains, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    : mpi_comm(mpi_comm)
    {
      private_init_domains(target_domains);
    }

    /**
     * DO NOT CALL THIS YOURSELF
     * this is here because KOKKOS_LAMBDAS cannot be declared in constructors or private methods with nvcc
     **/
    void private_init_domains( const Kokkos::View< int* > domains );
    
    /**
     * Create a new ViewCommunicator by listing local object to send to each remote domain (map<int, vector> representation).
     * Same object can be sent to multiple domains
     * 
     * @param ghost_map a 'rank -> local octant list' map to describe which objects to send to each MPI process
     **/
    ViewCommunicator( const std::map<int, std::vector<uint32_t>>& ghost_map, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() );   
    
    /**
     * Create a new ViewCommunicator by listing local object to send to each remote domain ( Kokkos::View representation )
     * Same object can be sent to multiple domains
     * 
     * @param send_sizes an array of size `MPI_comm_size()` describing how many object to send to each rank
     * @param send_iObj local object indices to send. First send_sizes(0) objects in send_iObj will be sent to rank 0, etc...
     **/
    ViewCommunicator( const Kokkos::View< uint32_t* > send_sizes, const Kokkos::View< uint32_t* > send_iObj, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    : mpi_comm(mpi_comm)
    {
      private_init_map(send_sizes, send_iObj);
    }

    /// Get the number of ghosts in local process with the current exchange pattern
    uint32_t getNumGhosts() const;

    /**
     * Generic function to send object data stored in U to Ughost according to communication pattern
     * 
     * @tparam is the Kokkos::View type. It must be Kokkos::LayoutLeft
     * @tparam distributed_dim position distributed objects index coordinate in U(..,..,...). 
     *         When distributed object index is not the leftmost index (distributed_dim != DataArray_t::rank-1), 
     *         packing/unpacking is less efficient because data needs to be transposed.
     * @param U a view containing data distributed accross the distributed_dim-th dimension. 
     *          Must be in accordance with the communication pattern.
     * @param Ughost A view to recieve the exchanged objects. This must be allocated to have getNumGhosts() 
     *               elements in the distributed_dim-th dimension, and must have the same shape than U in all other dimensions
     **/
    template< int distributed_dim, typename DataArray_t >
    void exchange_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const;

    /**
     * Generic function to reduce ghost values to source object
     * Ghost values from every process are sent to owning process and then added to the local non-ghost cell
     * 
     * @tparam is the Kokkos::View type. It must be Kokkos::LayoutLeft
     * @tparam distributed_dim position distributed objects index coordinate in U(..,..,...). 
     *         When distributed object index is not the leftmost index (distributed_dim != DataArray_t::rank-1), 
     *         packing/unpacking is less efficient because data needs to be transposed.
     * @param U a view containing data distributed accross the distributed_dim-th dimension. 
     *          Must be in accordance with the communication pattern.
     * @param Ughost A view to recieve the exchanged objects. This must have getNumGhosts() 
     *               elements in the distributed_dim-th dimension, and must have the same shape than U in all other dimensions
     **/
    template< int distributed_dim, typename DataArray_t >
    void reduce_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const;

private:
    void private_init_map( const Kokkos::View< uint32_t* > send_sizes, const Kokkos::View< uint32_t* > send_iOcts );

    Kokkos::View<uint32_t*> recv_sizes, send_sizes; //!Number of octants to send/recv for each proc
    Kokkos::View<uint32_t*>::HostMirror recv_sizes_host, send_sizes_host; //!Number of octants to send/recv for each proc
    Kokkos::View<uint32_t*> send_iOcts; //! List of octants to send (first send_sizes[0] iOcts to send to rank[0] and so on...)
    uint32_t nbghosts_recv;
    MpiComm mpi_comm;    
};

namespace ViewCommunicator_impl{
  /***
   * Note : Implementation detail documentation use octants as distributed objects
   * this is from a time when 
   ***/

  using namespace userdata_utils;  

  template< typename Layout_t >
  Kokkos::LayoutLeft to_LayoutLeft( const Layout_t& l )
  {
    Kokkos::LayoutLeft ll(l.dimension[0],
                          l.dimension[1],
                          l.dimension[2],
                          l.dimension[3],
                          l.dimension[4],
                          l.dimension[5],
                          l.dimension[6],
                          l.dimension[7]
                        );
    return ll;
  };

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

  template <typename PackBuffer_t, int iOct_pos, typename DataArray_t>
  PackBuffer_t allocate_packed( const DataArray_t& U, uint32_t send_oct_count, const Kokkos::View<uint32_t*>::HostMirror& send_sizes_host )
  {
    // When iOct is the rightmost subscript in U, a unique view of size (:,...,sum(send_sizes_host)) is 
    // allocated,
    // When iOct is not the rightmost subscript in U, a temporary transposed array is created

    static_assert( std::is_same<typename PackBuffer_t::array_layout, Kokkos::LayoutLeft>::value, 
                 "ViewCommunicator only supports Kokkos::LayoutLeft PackBuffer" ); 

    constexpr int dim = (int)DataArray_t::rank;
    
    // Allocate send_buffers with same dimension for each octant but with sum(send_sizes_host) octants
    // iOct is also displaced to the rightmost coordinate (if it's not already the case)
    Kokkos::LayoutLeft extents_send = to_LayoutLeft(U.layout());
    for(int i=iOct_pos; i<dim-1; i++)
      extents_send.dimension[i] = extents_send.dimension[i+1];
    extents_send.dimension[dim-1] = send_oct_count; 
    PackBuffer_t send_buffers("Send buffers", extents_send); 

    return send_buffers;
  }

  /**
   * Copy Octants `send_iOcts` from U to the MPI buffers
   * @tparam iOct_pos position of coorinate iOct in U
   * @returns the list of buffers ready to be sent to every rank, buffers are transposed from U to have iOct as leftmost subscript
   **/
  template <typename MPIBuffer_t, typename PackBuffer_t, int iOct_pos, typename DataArray_t>
  std::vector<MPIBuffer_t> pack( const DataArray_t& U, const Kokkos::View<uint32_t*>& send_iOcts, const Kokkos::View<uint32_t*>::HostMirror& send_sizes_host )
  {
    // When iOct is the rightmost subscript in U, a unique view of size (:,...,sum(send_sizes_host)) is 
    // allocated and then sliced in subviews for each rank,
    // When iOct is not the rightmost subscript in U, a temporary transposed array is created

    constexpr int dim = (int)DataArray_t::rank;
    
    // Allocate send_buffers with same dimension for each octant but with sum(send_sizes_host) octants
    // iOct is also displaced to the rightmost coordinate (if it's not already the case)
    PackBuffer_t send_buffers = allocate_packed<PackBuffer_t, iOct_pos>(U, send_iOcts.size(), send_sizes_host);

    uint32_t elts_per_octs = 1;
    for(int i=0; i<dim; i++)
      if( i!= iOct_pos )
        elts_per_octs *= U.extent(i);

    // Copy values to send from U to send_buffers
    Kokkos::parallel_for( "ViewCommunicator::fill_send_buffer", send_buffers.size(),
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
  template <int iOct_pos, typename DataArray_right_t, typename DataArray_t>
  std::enable_if_t< iOct_pos < DataArray_t::rank-1 , 
  void > transpose( const DataArray_right_t& Ughost_right_iOct, const DataArray_t& Ughost )
  {
    // When iOct is not the rightmost index, a temportary MPI buffer 
    // with iOct rightmost index is used and has to be transposed

    constexpr int rank = DataArray_t::rank();

    // Verify Ghost allocation has the right size
    DYABLO_ASSERT_HOST_RELEASE( Ughost.extent(iOct_pos) == Ughost_right_iOct.extent(rank-1),
      "Ughost is not allocated to the expected size" );

    uint32_t elts_per_octs = octant_size<DataArray_t, iOct_pos>(Ughost);

    // Transpose value from Ughost_right_iOct to Ughost
    Kokkos::parallel_for( "ViewCommunicator::unpack_transpose", Ughost_right_iOct.size(),
                          KOKKOS_LAMBDA(uint32_t index)
    {
      uint32_t iOct = index/elts_per_octs;
      uint32_t i = index%elts_per_octs;
      
      get_U<iOct_pos>(Ughost, iOct, i) = get_U<rank-1>(Ughost_right_iOct, iOct, i);
    });
  }

  /**
   * Transfert values from Ughost_right_iOct to Ughost
   * When iOct is rightmost index un Ughost there is nothing to transpose 
   **/
  template <int iOct_pos, typename DataArray_right_t, typename DataArray_t>
  std::enable_if_t< iOct_pos == DataArray_t::rank-1 , 
  void > transpose( const DataArray_right_t& Ughost_right_iOct, const DataArray_t& Ughost )
  {
    Kokkos::deep_copy( Ughost, Ughost_right_iOct );
  }

} // namespace ViewCommunicator_impl

template< int iOct_pos, typename DataArray_t >
void ViewCommunicator::exchange_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const
{ 
  using namespace ViewCommunicator_impl;
  using MPI_Request_t = MpiComm::MPI_Request_t;

  using PackBuffer = Kokkos::View< typename DataArray_t::data_type, Kokkos::LayoutLeft >;
#ifdef MPI_IS_CUDA_AWARE    
  using MPIBuffer = PackBuffer;
#else
  using MPIBuffer = typename PackBuffer::HostMirror;
#endif

  DYABLO_ASSERT_HOST_RELEASE( Ughost.extent(iOct_pos) == nbghosts_recv, "Mismatch between view extent and expected ghost count" );

  int nb_proc = mpi_comm.MPI_Comm_size();

  // Pack send buffers from U, allocate recieve buffers
  std::vector<MPIBuffer> send_buffers = pack<MPIBuffer, PackBuffer, iOct_pos>( U, this->send_iOcts, this->send_sizes_host );

  // Allocate Ughost_tmp with same volume of data, but with iOct at rightmost position
  Kokkos::LayoutLeft extents_Ughost_tmp = to_LayoutLeft(U.layout());
  for(uint32_t i=iOct_pos; i<DataArray_t::rank-1; i++)
    extents_Ughost_tmp.dimension[i] = extents_Ughost_tmp.dimension[i+1];
  extents_Ughost_tmp.dimension[DataArray_t::rank-1] = this->nbghosts_recv;
  PackBuffer Ughost_tmp(U.label()+"_ghost", extents_Ughost_tmp);

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

template <typename MPIBuffer_t, typename PackBuffer_t, int iOct_pos, typename DataArray_t>
std::vector<MPIBuffer_t> pack_ghosts( const DataArray_t& Ughost, const Kokkos::View<uint32_t*>::HostMirror& ghost_send_sizes_host )
{
  using namespace ViewCommunicator_impl;

  // When iOct is the rightmost subscript in U, a unique view of size (:,...,sum(send_sizes_host)) is 
  // allocated and then sliced in subviews for each rank,
  // When iOct is not the rightmost subscript in U, a temporary transposed array is created

  constexpr int dim = (int)DataArray_t::rank;
  
  // Allocate send_buffers with same dimension for each octant but with sum(send_sizes_host) octants
  // iOct is also displaced to the rightmost coordinate (if it's not already the case)
  PackBuffer_t send_buffers = allocate_packed<PackBuffer_t, iOct_pos>(Ughost, Ughost.extent(iOct_pos), ghost_send_sizes_host);

  uint32_t elts_per_octs = 1;
  for(int i=0; i<dim; i++)
    if( i!= iOct_pos )
      elts_per_octs *= Ughost.extent(i);

  // Copy values to send from U to send_buffers
  Kokkos::parallel_for( "ViewCommunicator::fill_send_buffer", send_buffers.size(),
                        KOKKOS_LAMBDA(uint32_t index)
  {
    uint32_t iGhost = index/elts_per_octs;
    uint32_t iOct_origin = iGhost;
    uint32_t i = index%elts_per_octs;
    
    // copy octant data with iOct dimension moved from iOct_pos to DataArray_t::rank-1
    get_U<DataArray_t::rank-1>(send_buffers, iGhost, i) = get_U<iOct_pos>(Ughost, iOct_origin, i);
  });

  // Slice send_buffers into subviews
  return get_subviews<MPIBuffer_t>(send_buffers, ghost_send_sizes_host);
}

template< int iOct_pos, typename DataArray_t >
void ViewCommunicator::reduce_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const
{
  using namespace ViewCommunicator_impl;
  using MPI_Request_t = MpiComm::MPI_Request_t;
  
  using PackBuffer = Kokkos::View< typename DataArray_t::data_type, Kokkos::LayoutLeft >;
  #ifdef MPI_IS_CUDA_AWARE    
    using MPIBuffer = PackBuffer;
  #else
    using MPIBuffer = typename PackBuffer::HostMirror;
  #endif

  int nb_proc = mpi_comm.MPI_Comm_size();

  // Send and recv buffers are reversed in this method compared to exchange_ghosts()
  // Send buffers are Ughost sliced into subviews of sizes *recv*_sizes_host[i]
  std::vector<MPIBuffer> send_buffers = pack_ghosts<MPIBuffer, PackBuffer, iOct_pos>(Ughost, recv_sizes_host);
  // Recv buffers are allocated and sliced into subviews of size *send*_sizes_host[i]
  PackBuffer recv_buffers_device = allocate_packed<PackBuffer,iOct_pos>( U, send_iOcts.size(), send_sizes_host );
  std::vector<MPIBuffer> recv_buffers = get_subviews<MPIBuffer>(recv_buffers_device, send_sizes_host);
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

  // Unpack recv buffers to recv_buffers_device 
  unpack(recv_buffers, recv_buffers_device, this->send_sizes_host);

  {
    constexpr int dim = DataArray_t::rank;
    uint32_t elts_per_octs = 1;
    for(int i=0; i<dim-1; i++)
        elts_per_octs *= recv_buffers_device.extent(i);

    auto& send_iOcts = this->send_iOcts;

    // Accumulate ghost cells gathered from all process in local cells
    Kokkos::parallel_for( "ViewCommunicator::reduce_ghosts::unpack_reduce", recv_buffers_device.size(),
      KOKKOS_LAMBDA (uint32_t index)
    {
      uint32_t iOct_ghost = index/elts_per_octs;
      uint32_t i = index%elts_per_octs;

      uint32_t iOct_local = send_iOcts(iOct_ghost);
      
      real_t& local_cell_value = get_U<iOct_pos>(U, iOct_local, i);
      real_t  ghost_cell_value = get_U<DataArray_t::rank-1>(recv_buffers_device, iOct_ghost, i);
      Kokkos::atomic_add( &local_cell_value, ghost_cell_value );
    });
  }

}

} // namespace dyablo