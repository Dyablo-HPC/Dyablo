#include "amr/AMRmesh_hashmap_new.h"

#include <Kokkos_UnorderedMap.hpp>
#include "morton_utils.h"
#include "mpi/GhostCommunicator.h"

namespace dyablo {

using markers_t = Kokkos::View<int*, AMRmesh_hashmap_new::Storage_t::MemorySpace>;

struct AMRmesh_hashmap_new::PData{
  markers_t markers;
  int level_max;
  bool distibuted_mesh = false; // true if load_balance have already been called at least once
  std::map<int, std::vector<oct_index_t>> local_octants_to_send; // Ghosts to send to each other process
};

AMRmesh_hashmap_new::AMRmesh_hashmap_new( int dim, int balance_codim, 
                      const std::array<bool,3>& periodic, 
                      uint8_t level_min, uint8_t level_max,
                      const MpiComm& mpi_comm)
: storage( dim, (mpi_comm.MPI_Comm_rank()==0)?1:0, 0 ),
  periodic({periodic[IX], periodic[IY], periodic[IZ]}),
  mpi_comm(mpi_comm),
  total_num_octs(1), first_local_oct(0),
  pdata(std::make_unique<PData>(PData{
    markers_t( "AMRmesh_hashmap_new::markers", 1 ),
    level_max
  }))
{}

AMRmesh_hashmap_new::~AMRmesh_hashmap_new()
{}

void AMRmesh_hashmap_new::adaptGlobalRefine()
{
  assert(!pdata->distibuted_mesh); // No GlobalRefine after first load balancing

  int ndim = getDim();
  int nbSubocts = 4*(ndim-1);
  oct_index_t old_nbOcts = getNumOctants();
  oct_index_t new_nbOcts = old_nbOcts*nbSubocts;

  Storage_t& old_storage = this->storage;
  LightOctree_storage<> old_storage_device( old_storage );
  LightOctree_storage<> new_storage_device( ndim, new_nbOcts, 0 );

  Kokkos::parallel_for( "AMRmesh_hashmap_new::adaptGlobalRefine", 
    old_nbOcts,
    KOKKOS_LAMBDA( oct_index_t iOct_old )
  {
    auto logical_coord = old_storage_device.get_logical_coords( {iOct_old, false} );
    level_t level_old = old_storage_device.getLevel( {iOct_old, false} );

    for(int dz=0; dz<(ndim-1); dz++)
      for(int dy=0; dy<2; dy++)
        for(int dx=0; dx<2; dx++)
        {
          // Compute iOct_new for suboctant using morton order
          oct_index_t iOct_new = iOct_old*nbSubocts + dx + 2*dy + 4*dz;
          logical_coord_t ix = 2*logical_coord[IX] + dx;
          logical_coord_t iy = 2*logical_coord[IY] + dy;
          logical_coord_t iz = 2*logical_coord[IZ] + dz;
          level_t level = level_old + 1;
          new_storage_device.set( {iOct_new, false}, ix, iy, iz, level );
        }
  });

  total_num_octs = total_num_octs*nbSubocts;
  old_storage = new_storage_device;
  pdata->markers = markers_t("AMRmesh_hashmap_new::markers", new_nbOcts);
}

void AMRmesh_hashmap_new::setMarker(uint32_t iOct, int marker)
{
  pdata->markers(iOct) = marker;
}

void AMRmesh_hashmap_new::adapt(bool dummy)
{
  #warning "TODO"
}

const std::map<int, std::vector<uint32_t>>& AMRmesh_hashmap_new::getBordersPerProc() const
{
  return pdata->local_octants_to_send;
}

namespace{

using morton_t = uint64_t;
using level_t = AMRmesh_hashmap_new::level_t;
using oct_index_t = AMRmesh_hashmap_new::oct_index_t;
using logical_coord_t = AMRmesh_hashmap_new::logical_coord_t;

KOKKOS_INLINE_FUNCTION
morton_t shift_level( const morton_t& m, int level_diff )
{
  if( level_diff >= 0 )
    return m << 3*level_diff;
  else
    return m >> -3*level_diff;
}

template <typename MemorySpace_t>
KOKKOS_INLINE_FUNCTION
morton_t compute_morton( const LightOctree_storage<MemorySpace_t>& octs, const AMRmesh_hashmap_new::oct_index_t& iOct_local, AMRmesh_hashmap_new::level_t target_level)
{
  auto pos = octs.get_logical_coords({iOct_local, false});
  AMRmesh_hashmap_new::level_t level = octs.getLevel({iOct_local, false});
  morton_t m = compute_morton_key( pos[IX], pos[IY], pos[IZ] );
  return shift_level( m, target_level-level );
}

// get first iOct_local with morton(iOct_local) >= morton
AMRmesh_hashmap_new::oct_index_t lower_bound_morton( const AMRmesh_hashmap_new::Storage_t& octs, morton_t morton, AMRmesh_hashmap_new::level_t target_level )
{
  using oct_index_t = AMRmesh_hashmap_new::oct_index_t;
  oct_index_t nbOcts = octs.getNumOctants();

  oct_index_t begin = 0;
  oct_index_t end = nbOcts-1;
  while( begin < end )
  {
    oct_index_t pivot = begin + (end-begin)/2;
    morton_t morton_pivot = compute_morton(octs, pivot, target_level);
    if( morton_pivot == morton )
      return pivot;
    else if( morton_pivot < morton )
      begin = pivot+1;
    else //if( morton_pivot > morton )
      end = pivot;
  }
  return begin;
}

struct NeighborPair
{
  oct_index_t iOct_local;
  int rank_neighbor;
};

std::map<int, std::vector<oct_index_t>> discover_ghosts(
  const AMRmesh_hashmap_new::Storage_t& octs_host,
  const std::vector<morton_t>& morton_intervals_,
  level_t level_max, 
  const Kokkos::Array<bool,3>& periodic,
  const MpiComm& mpi_comm)
{
  int mpi_rank = mpi_comm.MPI_Comm_rank();
  int mpi_size = mpi_comm.MPI_Comm_size();
  int ndim = octs_host.getNdim();

  LightOctree_storage<> storage_device = octs_host;

  // Copy morton_intervals to device
  Kokkos::View<morton_t*> morton_intervals_device("discover_ghosts::morton_intervals", morton_intervals_.size());
  {
    auto morton_intervals_host = Kokkos::create_mirror_view( morton_intervals_device  );
    std::copy( morton_intervals_.begin(), morton_intervals_.end(), morton_intervals_host.data() );
    Kokkos::deep_copy( morton_intervals_device, morton_intervals_host );
  }
  
  
  size_t neighbor_count_guess = storage_device.getNumOctants();
  Kokkos::UnorderedMap< NeighborPair, void > neighborMap( neighbor_count_guess );

  // Storage must be allocated before launching the kernel, 
  // but we don't know the number of ghosts to send :
  // When insert fails, we reallocate and restart from the first failed insert
  oct_index_t first_fail = 0;
  while( first_fail < storage_device.getNumOctants() )
  {
    oct_index_t iter_start = first_fail;
    first_fail = storage_device.getNumOctants();
    Kokkos::parallel_reduce( "discover_ghosts", 
      Kokkos::RangePolicy<>( iter_start, storage_device.getNumOctants()),
      KOKKOS_LAMBDA( oct_index_t iOct, oct_index_t& first_fail_local )
    {
      auto compute_morton = [&]( const Kokkos::Array<logical_coord_t, 3>& pos, level_t level )
      { 
        morton_t morton = compute_morton_key( pos[IX], pos[IY], pos[IZ] );
        morton = shift_level( morton, level_max-level );
        return morton;
      };

      auto find_rank = [&]( morton_t morton )
      { 
        // upper_bound : first verifying value > morton
        int rank;
        {
          int begin = 0;
          int end = mpi_size;
          while( begin < end )
          {
            int pivot = begin + (end-begin)/2;
            morton_t morton_pivot = morton_intervals_device(pivot);
            if( morton_pivot <= morton )
              begin = pivot+1;
            else //if( morton_pivot > morton )
              end = pivot;
          }
          // Just before the upper bound
          rank=begin-1;
        }
        assert( rank<mpi_size );
        assert( morton_intervals_device(rank) <= morton);
        assert( morton_intervals_device(rank+1) > morton);
        return rank;
      };

      // Register current octant as ghost for neighbor_rank
      auto register_neighbor = [&]( int neighbor_rank )
      {
        if(neighbor_rank != mpi_rank)
        {
          auto insert_result = neighborMap.insert( NeighborPair{iOct, neighbor_rank } );
          if( insert_result.failed() )
            first_fail_local = first_fail_local < iOct ? first_fail_local : iOct;
        }
      };

      Kokkos::Array<logical_coord_t, 3> pos = storage_device.get_logical_coords( {iOct, false} );
      level_t level = storage_device.getLevel( {iOct, false} );

      assert(find_rank(compute_morton(pos, level)) == mpi_rank);

      logical_coord_t max_i = storage_device.cell_count(level);
      int dz_max = (ndim == 2)? 0:1;
      for( int dz=-dz_max; dz<=dz_max; dz++ )
      for( int dy=-1; dy<=1; dy++ )
      for( int dx=-1; dx<=1; dx++ )
      if(   (dx!=0 || dy!=0 || dz!=0)
          && (periodic[IX] || ( 0<=pos[IX]+dx && pos[IX]+dx<max_i ))
          && (periodic[IY] || ( 0<=pos[IY]+dy && pos[IY]+dy<max_i ))
          && (periodic[IZ] || ( 0<=pos[IZ]+dz && pos[IZ]+dz<max_i )) )
      {
        Kokkos::Array<logical_coord_t, 3> pos_n{
          (pos[IX]+dx+max_i)%max_i, 
          (pos[IY]+dy+max_i)%max_i, 
          (pos[IZ]+dz+max_i)%max_i
        };
        morton_t morton_n = compute_morton( pos_n, level );
        int neighbor_rank = find_rank( morton_n );

        // Verify that the whole same-size virtual neighbor is owned by neighbor_rank
        // i.e : last suboctant of same-size neighbor is owned by the same MPI
        // TODO : get morton of last neighbor to filter even more
        morton_t morton_next = shift_level(morton_n, level-level_max) + 1;
                morton_next = shift_level(morton_next, level_max-level);
        if( level<level_max && morton_next < morton_intervals_device(neighbor_rank+1) )
        {// Neighbors are owned by only one MPI
          register_neighbor(neighbor_rank);
        }
        else
        {// Neighbors may be scattered between multiple MPIs  
          
          // Apply offset to get smaller origin neighbor
          Kokkos::Array<logical_coord_t, 3> pos_n_smaller_origin{
            (pos_n[IX] << 1) + (dx == -1), // add one level + right half of suboctant if left of original cell
            (pos_n[IY] << 1) + (dy == -1),
            (pos_n[IZ] << 1) + (dz == -1),
          };

          // Iterate over neighbor suboctants
          int sx_max = (dx==0); // constrained to the same plane as origin if offset in this direction
          int sy_max = (dy==0);
          int sz_max = (ndim==2) ? 0 : (dz==0);
          for( int16_t sz=0; sz<=sz_max; sz++ )
          for( int16_t sy=0; sy<=sy_max; sy++ )
          for( int16_t sx=0; sx<=sx_max; sx++ )
          {
            Kokkos::Array<logical_coord_t, 3> pos_n_smaller{
              pos_n_smaller_origin[IX] + sx,
              pos_n_smaller_origin[IY] + sy,
              pos_n_smaller_origin[IZ] + sz,
            };
            morton_t m_suboctant = compute_morton( pos_n_smaller, level+1 );
            int neighbor_rank = find_rank(m_suboctant);
            register_neighbor(neighbor_rank);
          }
        }
      }
    }, Kokkos::Min<oct_index_t>(first_fail) ); 
  
    // first_fail should be std::numeric_limit<oct_index_t>::max() if every insert succeeded
    if( first_fail < storage_device.getNumOctants() )
    {
      std::cout << "Ghost storage too small : rehash" << std::endl;
      neighbor_count_guess *= 2;
      neighborMap.rehash( neighbor_count_guess );
      std::cout << "Restart from iOct " << first_fail << std::endl;
    }
  }

  // Copy content of neighborMap to result variable
  std::map<int, std::vector<oct_index_t>> to_send;
  {
    // Compute number of neighbors to send to each other rank
    Kokkos::View<oct_index_t*> to_send_count_device("discover_ghosts::to_send_count", mpi_size);
    Kokkos::parallel_for( "discover_ghosts::count_neighbors", neighborMap.capacity(),
      KOKKOS_LAMBDA( uint32_t i )
    {
      if( neighborMap.valid_at(i) )
      {
        const NeighborPair& p = neighborMap.key_at(i);
        Kokkos::atomic_increment( &to_send_count_device(p.rank_neighbor) );
      }
    });

    // Compute offsets marking beginning of each process in a single contiguous list of neighbors
    Kokkos::View<oct_index_t*> to_send_offset_device("discover_ghosts::to_send_offset", mpi_size);
    oct_index_t to_send_count_total = 0; // Total number of neighbor octants
    Kokkos::parallel_scan( "discover_ghosts::compute_offsets", mpi_size,
      KOKKOS_LAMBDA( int rank, oct_index_t& offset_local, bool final )
    {
      if(final)
        to_send_offset_device(rank) = offset_local;
      offset_local += to_send_count_device(rank);
    }, to_send_count_total);

    // Copy sizes and offsets to host to allocate std::vectors
    auto to_send_count_host = Kokkos::create_mirror_view( to_send_count_device );
    auto to_send_offset_host = Kokkos::create_mirror( to_send_offset_device ); // Makes a copy
    Kokkos::deep_copy(to_send_count_host, to_send_count_device);
    Kokkos::deep_copy(to_send_offset_host, to_send_offset_device);
    
    // Fill to_send_device with neighbors to send to other ranks
    Kokkos::View<oct_index_t*> to_send_device("discover_ghosts::to_send", to_send_count_total);
    Kokkos::parallel_for( "fill_neighbors", neighborMap.capacity(),
      KOKKOS_LAMBDA( uint32_t i )
    {
      if( neighborMap.valid_at(i) )
      {
        const NeighborPair& p = neighborMap.key_at(i);
        oct_index_t offset = Kokkos::atomic_fetch_add( &to_send_offset_device(p.rank_neighbor), 1 );
        to_send_device( offset ) = p.iOct_local;
      }
    });

    // Copy from Kokkos::View to std::vector for return
    for(int rank=0; rank<mpi_size; rank++)
    {
      oct_index_t rank_size = to_send_count_host(rank);
      oct_index_t view_offset = to_send_offset_host(rank);

      auto to_send_subview_device = Kokkos::subview( to_send_device, std::make_pair(view_offset, view_offset+rank_size) );
      auto to_send_host = Kokkos::create_mirror_view( to_send_subview_device );
      Kokkos::deep_copy( to_send_host, to_send_subview_device );

      to_send[ rank ] = std::vector<oct_index_t>(to_send_host.data(), to_send_host.data()+rank_size);
    }
    
  }

  return to_send;  
}

} // namespace

std::map<int, std::vector<AMRmesh_hashmap_new::oct_index_t>> 
AMRmesh_hashmap_new::loadBalance(level_t level)
{
    int mpi_rank = this->getMpiComm().MPI_Comm_rank();
    int mpi_size = this->getMpiComm().MPI_Comm_size();
    level_t level_max = pdata->level_max;

    std::vector<morton_t> new_morton_intervals(mpi_size+1);    
    // Get evenly distributed initial intervals and gather mortons
    {
      int nb_mortons = 0;
      global_oct_index_t iOct_begin = this->getGlobalIdx( 0 );
      global_oct_index_t iOct_end = this->getGlobalIdx( this->getNumOctants() );
      for(int i=0; i<mpi_size; i++)
      {
          global_oct_index_t idx = (this->getGlobalNumOctants()*i)/mpi_size ;
          // For each ixd inside old domain, compute morton
          // and fill morton_intervals for this rank
          if( iOct_begin <= idx && idx < iOct_end )
          {
              new_morton_intervals[i] = compute_morton( storage, idx-iOct_begin, level_max );
              nb_mortons++;
          }
      }   

      // allgather morton_intervals
      mpi_comm.MPI_Allgatherv_inplace( new_morton_intervals.data(), nb_mortons );
      new_morton_intervals[0] = 0;
      new_morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
      for(int rank=0; rank<mpi_size; rank++)
      {
          // Truncate suboctants to keep `level` levels of suboctants compact
          new_morton_intervals[rank] = (new_morton_intervals[rank] >> (3*level)) << (3*level);
      }
      assert(new_morton_intervals[mpi_rank] <= new_morton_intervals[mpi_rank+1] );
    }

    std::cout << "Rank " << mpi_rank << ": new morton interval [" << new_morton_intervals[mpi_rank] << ", " << new_morton_intervals[mpi_rank+1] << "[" << std::endl;

    // Compute `new_oct_intervals` corresponding to `morton_intervals`
    std::vector<global_oct_index_t> new_oct_intervals(mpi_size+1); // First global index for rank i
    {
        morton_t old_morton_interval_begin, old_morton_interval_end;
        {
          if( storage.getNumOctants() != 0 )
            old_morton_interval_begin = compute_morton( storage, 0, level_max );
          else
            old_morton_interval_begin = 0;
          std::vector<morton_t> old_morton_intervals(mpi_size+1);
          // TODO maybe just send to mpi_rank-1 ?
          mpi_comm.MPI_Allgather( &old_morton_interval_begin, old_morton_intervals.data(), 1);
          old_morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
          for(int rank=mpi_size-1; rank>0; rank--)
            if( old_morton_intervals[rank] == 0 )
              old_morton_intervals[rank] = old_morton_intervals[rank+1];
          old_morton_interval_begin = old_morton_intervals[mpi_rank];
          old_morton_interval_end = old_morton_intervals[mpi_rank+1];
        }            

        std::cout << "Rank " << mpi_rank << ": old morton interval [" << old_morton_interval_begin << ", " << old_morton_interval_end << "[" << std::endl;

        int nb_local_pivots=0;
        for(int rank=0; rank<mpi_size; rank++)
        {
            if( old_morton_interval_begin <= new_morton_intervals[rank] && new_morton_intervals[rank] < old_morton_interval_end ) // Determine if pivot is inside of local process
            {
                // find first local octant with morton >= morton_interval[rank]
                oct_index_t pivot = lower_bound_morton( storage, new_morton_intervals[rank], level_max );
                nb_local_pivots++;
                new_oct_intervals[rank] = this->getGlobalIdx(pivot);
            }
        }

        mpi_comm.MPI_Allgatherv_inplace( new_oct_intervals.data(), nb_local_pivots );
        new_oct_intervals[mpi_size] = this->getGlobalNumOctants();

    }
    std::cout << "Rank " << mpi_rank << ": iOct interval [" << new_oct_intervals[mpi_rank] << ", " << new_oct_intervals[mpi_rank+1] << "[" << std::endl;
    assert( new_oct_intervals[mpi_rank] <= new_oct_intervals[mpi_rank+1] );

    // List octants to exchange
    std::map<int, std::vector<oct_index_t>> loadbalance_to_send;
    for( int rank=0; rank<mpi_size; rank++ )
    {
      // intersection between local and remote ranks
      global_oct_index_t global_local_begin = this->getGlobalIdx(0);
      global_oct_index_t global_local_end = this->getGlobalIdx(this->getNumOctants()) ;
      global_oct_index_t global_intersect_begin = std::max( new_oct_intervals[rank],   global_local_begin );
      global_oct_index_t global_intersect_end   = std::min( new_oct_intervals[rank+1], global_local_end  );

      if( global_intersect_begin < global_intersect_end )
      {
        oct_index_t to_send_begin = global_intersect_begin - global_local_begin;
        oct_index_t to_send_end = global_intersect_end - global_local_begin;
        assert( to_send_begin < this->getNumOctants() );
        assert( to_send_end <= this->getNumOctants() );
        std::vector<oct_index_t> to_send_rank(to_send_end-to_send_begin);
        for(oct_index_t i=0; i<to_send_end-to_send_begin; i++)
        {
            to_send_rank[i] = i+to_send_begin;
        }
        loadbalance_to_send[rank] = to_send_rank;
      }
    }

    // Exchange octs that changed domain 
    oct_index_t new_nbOcts = new_oct_intervals[mpi_rank+1]-new_oct_intervals[mpi_rank];
    LightOctree_storage<> new_storage_device(this->getDim(), new_nbOcts, 0);
    {
      // Use storage on device to perform remaining operations
      LightOctree_storage<> old_storage_device = storage;

      GhostCommunicator_kokkos loadbalance_communicator( loadbalance_to_send );
      loadbalance_communicator.exchange_ghosts( old_storage_device.oct_data, new_storage_device.oct_data );

      assert( new_storage_device.oct_data.extent(0) == new_nbOcts );
    }

    // update misc metadata
    this->first_local_oct = new_oct_intervals[mpi_rank];
    pdata->distibuted_mesh = true;   

    pdata->local_octants_to_send = discover_ghosts(new_storage_device, new_morton_intervals, level_max, this->periodic, mpi_comm);

    // Raw view for ghosts
    LightOctree_storage<>::oct_data_t neighbor_octs_device;
    GhostCommunicator_kokkos ghost_comm( pdata->local_octants_to_send  );
    ghost_comm.exchange_ghosts( new_storage_device.oct_data, neighbor_octs_device );

    {
      int ndim = new_storage_device.getNdim();
      //oct_index_t new_nbOcts = new_storage_device.getNumOctants();
      oct_index_t new_nbGhosts = neighbor_octs_device.extent(0);

      // We need to go through a temporary device storage because
      // subviews are non-contiguous and deep_copy is only supported in same memory space
      LightOctree_storage<> storage_device(ndim, new_nbOcts, new_nbGhosts );
      Kokkos::deep_copy( storage_device.getLocalSubview(), new_storage_device.getLocalSubview() );
      Kokkos::deep_copy( storage_device.getGhostSubview(), neighbor_octs_device );

      // Reallocate storage with new size
      Storage_t& this_storage = this->storage;
      this_storage = storage_device;
    }

    pdata->markers = markers_t("markers", this->getNumOctants());

    // Print new domain decomposition
    // TODO clean raw console outputs?
    if( new_nbOcts != 0 )
    {
      std::cout << "Rank " << mpi_rank << ": actual morton interval [" << compute_morton( storage, 0, level_max) << ", " << compute_morton( storage,  getNumOctants()-1, level_max) << "]" << std::endl;
      assert( compute_morton( storage, 0, level_max) >= new_morton_intervals[mpi_rank] );
      assert( compute_morton( storage, getNumOctants()-1, level_max) < new_morton_intervals[mpi_rank+1] );
    }
    else 
    {
      std::cout << "Rank " << mpi_rank << ": actual morton interval [EMPTY]" << std::endl;
      std::cout << "WARNING : Rank has 0 octant, this is probably not okay" << std::endl;
    } 

    assert(this->getNumOctants() > 0); // Process cannot be empty

    return loadbalance_to_send;
}

void AMRmesh_hashmap_new::loadBalance_userdata( level_t compact_levels, DataArrayBlock& userData )
{
  #warning "TODO"
}




} // namespace dyablo