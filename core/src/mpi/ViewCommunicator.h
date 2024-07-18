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

#ifdef DYABLO_COMPILE_PABLO
    static ViewCommunicator from_mesh( const AMRmesh_pablo& mesh, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    {
      return ViewCommunicator( mesh.getBordersPerProc(), mpi_comm );
    }
#endif

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
   **/
  template <typename PackBuffer_t>
  std::vector<PackBuffer_t> get_subviews(const PackBuffer_t& view, const Kokkos::View<uint32_t*>::HostMirror& sizes)
  {
    int nb_proc = sizes.size();

    static_assert( std::is_same<typename PackBuffer_t::array_layout, Kokkos::LayoutLeft>::value, 
                 "PackBuffer must be LayoutLeft to get subviews" ); 
    
    std::vector<PackBuffer_t> res(nb_proc);
    uint32_t iOct_offset = 0;
    for(int rank=0; rank<nb_proc; rank++)
    {
      uint32_t iOct_range_end = iOct_offset+sizes(rank);
      // This only works if PackBuffer_t is LayoutLeft
      // Otherwise, subview is not contiguous and this is necessary for MPI
      res[rank] = get_subview( view, iOct_offset, iOct_range_end);

      iOct_offset += sizes(rank);
    }

    Kokkos::fence();

    return res;
  }
} // namespace ViewCommunicator_impl

template< int iOct_pos, typename DataArray_t >
void ViewCommunicator::exchange_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const
{ 
  using namespace ViewCommunicator_impl;
  using MPI_Request_t = MpiComm::MPI_Request_t;

  using PackBuffer = Kokkos::View< typename DataArray_t::data_type, Kokkos::LayoutLeft >;
  
#ifdef MPI_IS_CUDA_AWARE    
  using MPIMemorySpace = typename DataArray_t::memory_space;
#else
  using MPIMemorySpace = Kokkos::HostSpace;
#endif
  
  using MPIBuffer = decltype( Kokkos::create_mirror_view( MPIMemorySpace(), std::declval<PackBuffer>() ) );

  DYABLO_ASSERT_HOST_RELEASE( Ughost.extent(iOct_pos) == nbghosts_recv, "Mismatch between view extent and expected ghost count" );

  constexpr int ndim = DataArray_t::rank();
  int nb_proc = mpi_comm.MPI_Comm_size();
  uint32_t elts_per_octs = 1;
  for( int i=0; i<ndim; i++ )
    if(i != iOct_pos)
      elts_per_octs *= U.extent(i);

  // Pack send buffers from U, allocate recieve buffers
  std::vector<PackBuffer> send_buffers;
  {
    // Allocate packbuffer with same dimension for each octant but with sum(send_sizes_host) octants
    // iOct is also displaced to the rightmost coordinate (if it's not already the case)
    size_t send_oct_count = send_iOcts.size();
    Kokkos::LayoutLeft extents_send = to_LayoutLeft(U.layout());
    for(int i=iOct_pos; i<ndim-1; i++)
      extents_send.dimension[i] = extents_send.dimension[i+1];
    extents_send.dimension[ndim-1] = send_oct_count; 
    PackBuffer packbuffer("send_buffers", extents_send);

    auto& send_iOcts = this->send_iOcts;

    // Copy values to send from U to packbuffer
    Kokkos::parallel_for( "ViewCommunicator::fill_send_buffer", packbuffer.size(),
                          KOKKOS_LAMBDA(uint32_t index)
    {
      uint32_t iGhost = index/elts_per_octs;
      uint32_t iOct_origin = send_iOcts(iGhost);
      uint32_t i = index%elts_per_octs;
      
      // copy octant data with iOct dimension moved from iOct_pos to DataArray_t::rank-1
      get_U<ndim-1>(packbuffer, iGhost, i) = get_U<iOct_pos>(U, iOct_origin, i);
    });

    // Slice send_buffers into subviews
    send_buffers = get_subviews(packbuffer, send_sizes_host);
  }

  PackBuffer unpack_buffer;
  std::vector<PackBuffer> recv_buffers;
  {
    // Allocate unpack_buffer with same volume of data, but with iOct at rightmost position
    Kokkos::LayoutLeft extents_unpack_buffer = to_LayoutLeft(U.layout());
    for(uint32_t i=iOct_pos; i<DataArray_t::rank-1; i++)
      extents_unpack_buffer.dimension[i] = extents_unpack_buffer.dimension[i+1];
    extents_unpack_buffer.dimension[DataArray_t::rank-1] = this->nbghosts_recv;
    // TODO : use Ughost when PackBuffer == DataArray_t
    unpack_buffer = PackBuffer("recv_buffers", extents_unpack_buffer);

    recv_buffers = get_subviews(unpack_buffer, recv_sizes_host);
  }
    
  {
    std::vector<MPIBuffer> mpi_send_buffers(nb_proc);
    std::vector<MPIBuffer> mpi_recv_buffers(nb_proc);
    std::vector<MPI_Request_t> mpi_requests;
    // Post MPI_Isends
    for(int rank=0; rank<nb_proc; rank++)
    {
      if( send_buffers[rank].size() > 0 )
      {
        MPIBuffer send_buffer_rank = Kokkos::create_mirror_view( MPIMemorySpace(), send_buffers[rank] );
        Kokkos::deep_copy( send_buffer_rank, send_buffers[rank] );
        mpi_send_buffers[rank] = send_buffer_rank;   
        MPI_Request_t r = mpi_comm.MPI_Isend( send_buffer_rank, rank, 0 );
        mpi_requests.push_back(r);
      }
    }
    // Post MPI_Irecv   
    for(int rank=0; rank<nb_proc; rank++)
    {
      if( recv_buffers[rank].size() > 0 )
      {
        MPIBuffer recv_buffer_rank = Kokkos::create_mirror_view( MPIMemorySpace(), recv_buffers[rank] );
        mpi_recv_buffers[rank] = recv_buffer_rank;  
        MPI_Request_t r = mpi_comm.MPI_Irecv( recv_buffer_rank, rank, 0 );
        mpi_requests.push_back(r);
      }
    }

    mpi_comm.MPI_Waitall(mpi_requests.size(), mpi_requests.data());

    for(int rank=0; rank<nb_proc; rank++)
    {
      if( recv_buffers[rank].size() > 0 )
      {
        Kokkos::deep_copy( recv_buffers[rank], mpi_recv_buffers[rank] );
      }
    }
  }

  // Unpack unpack_buffer to Ughost
  Kokkos::parallel_for( "ViewCommunicator::unpack", unpack_buffer.size(),
                        KOKKOS_LAMBDA(uint32_t index)
  {
    uint32_t iOct = index/elts_per_octs;
    uint32_t i = index%elts_per_octs;
    
    get_U<iOct_pos>(Ughost, iOct, i) = get_U<ndim-1>(unpack_buffer, iOct, i);
  });
}

template< int iOct_pos, typename DataArray_t >
void ViewCommunicator::reduce_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const
{
  using namespace ViewCommunicator_impl;
  using MPI_Request_t = MpiComm::MPI_Request_t;

  using PackBuffer = Kokkos::View< typename DataArray_t::data_type, Kokkos::LayoutLeft >;
  
#ifdef MPI_IS_CUDA_AWARE    
  using MPIMemorySpace = typename DataArray_t::memory_space;
#else
  using MPIMemorySpace = Kokkos::HostSpace;
#endif
  
  using MPIBuffer = decltype( Kokkos::create_mirror_view( MPIMemorySpace(), std::declval<PackBuffer>() ) );

  DYABLO_ASSERT_HOST_RELEASE( Ughost.extent(iOct_pos) == nbghosts_recv, "Mismatch between view extent and expected ghost count" );

  constexpr int ndim = DataArray_t::rank();
  int nb_proc = mpi_comm.MPI_Comm_size();
  uint32_t elts_per_octs = 1;
  for( int i=0; i<ndim; i++ )
    if(i != iOct_pos)
      elts_per_octs *= U.extent(i);

  // Send and recv buffers are reversed in this method compared to exchange_ghosts()

  // Pack send buffers from Ughost
  // Send buffers are Ughost (eventually transposed) sliced into subviews of sizes *recv*_sizes_host[i]
  std::vector<PackBuffer> send_buffers;
  {
    // Allocate packbuffer with same dimension for each octant but with sum(*recv*_sizes_host) octants
    // iOct is also displaced to the rightmost coordinate (if it's not already the case)
    size_t send_oct_count = nbghosts_recv;
    Kokkos::LayoutLeft extents_send = to_LayoutLeft(Ughost.layout());
    for(int i=iOct_pos; i<ndim-1; i++)
      extents_send.dimension[i] = extents_send.dimension[i+1];
    extents_send.dimension[ndim-1] = send_oct_count; 
    PackBuffer packbuffer("send_buffers", extents_send);

    // Copy values to send from Ughost to packbuffer
    // TODO : use Ughost directly for send buffer if iOct is righmost index
    Kokkos::parallel_for( "ViewCommunicator::reduce::fill_send_buffer", packbuffer.size(),
                          KOKKOS_LAMBDA(uint32_t index)
    {
      uint32_t iGhost = index/elts_per_octs;
      uint32_t iOct_origin = iGhost;
      uint32_t i = index%elts_per_octs;
      
      // copy octant data with iOct dimension moved from iOct_pos to DataArray_t::rank-1
      get_U<ndim-1>(packbuffer, iGhost, i) = get_U<iOct_pos>(Ughost, iOct_origin, i);
    });

    // Slice send_buffers into subviews
    send_buffers = get_subviews(packbuffer, recv_sizes_host);
  }

  // Recv buffers are allocated and sliced into subviews of size *send*_sizes_host[i]
  PackBuffer unpack_buffer;
  std::vector<PackBuffer> recv_buffers;
  {
    // Allocate unpack_buffer with same volume of data, but with iOct at rightmost position
    Kokkos::LayoutLeft extents_unpack_buffer = to_LayoutLeft(U.layout());
    for(uint32_t i=iOct_pos; i<DataArray_t::rank-1; i++)
      extents_unpack_buffer.dimension[i] = extents_unpack_buffer.dimension[i+1];
    extents_unpack_buffer.dimension[DataArray_t::rank-1] = send_iOcts.size();
    // TODO : use Ughost when PackBuffer == DataArray_t
    unpack_buffer = PackBuffer("recv_buffers", extents_unpack_buffer);

    recv_buffers = get_subviews(unpack_buffer, send_sizes_host);
  }
  
  {
    std::vector<MPIBuffer> mpi_send_buffers(nb_proc);
    std::vector<MPIBuffer> mpi_recv_buffers(nb_proc);
    std::vector<MPI_Request_t> mpi_requests;
    // Post MPI_Isends
    for(int rank=0; rank<nb_proc; rank++)
    {
      if( send_buffers[rank].size() > 0 )
      {
        MPIBuffer send_buffer_rank = Kokkos::create_mirror_view( MPIMemorySpace(), send_buffers[rank] );
        Kokkos::deep_copy( send_buffer_rank, send_buffers[rank] );
        mpi_send_buffers[rank] = send_buffer_rank;   
        MPI_Request_t r = mpi_comm.MPI_Isend( send_buffer_rank, rank, 0 );
        mpi_requests.push_back(r);
      }
    }
    // Post MPI_Irecv   
    for(int rank=0; rank<nb_proc; rank++)
    {
      if( recv_buffers[rank].size() > 0 )
      {
        MPIBuffer recv_buffer_rank = Kokkos::create_mirror_view( MPIMemorySpace(), recv_buffers[rank] );
        mpi_recv_buffers[rank] = recv_buffer_rank;  
        MPI_Request_t r = mpi_comm.MPI_Irecv( recv_buffer_rank, rank, 0 );
        mpi_requests.push_back(r);
      }
    }

    mpi_comm.MPI_Waitall(mpi_requests.size(), mpi_requests.data());

    for(int rank=0; rank<nb_proc; rank++)
    {
      if( recv_buffers[rank].size() > 0 )
      {
        Kokkos::deep_copy( recv_buffers[rank], mpi_recv_buffers[rank] );
      }
    }
  }

  {
    auto& send_iOcts = this->send_iOcts;

    // Accumulate ghost cells gathered from all process in local cells
    Kokkos::parallel_for( "ViewCommunicator::reduce_ghosts::unpack_reduce", unpack_buffer.size(),
      KOKKOS_LAMBDA (uint32_t index)
    {
      uint32_t iOct_ghost = index/elts_per_octs;
      uint32_t i = index%elts_per_octs;

      uint32_t iOct_local = send_iOcts(iOct_ghost);
      
      real_t& local_cell_value = get_U<iOct_pos>(U, iOct_local, i);
      real_t  ghost_cell_value = get_U<DataArray_t::rank-1>(unpack_buffer, iOct_ghost, i);
      Kokkos::atomic_add( &local_cell_value, ghost_cell_value );
    });
  }
}

} // namespace dyablo