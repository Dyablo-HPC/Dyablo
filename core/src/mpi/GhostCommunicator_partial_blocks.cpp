#include "GhostCommunicator_partial_blocks.h"

namespace dyablo {

void precompute_facemask_cells( uint32_t bx, uint32_t by, uint32_t bz, uint32_t ghost_count, 
                                Kokkos::View<uint32_t*>& facemask_count, Kokkos::View<uint32_t**>& facemask_iCells )
{
  using CellMask = AMRmesh_hashmap_new::GhostMap_t::CellMask;
  using Face = AMRmesh_hashmap_new::GhostMap_t::Face;

  auto facemask_count_host = Kokkos::create_mirror_view(facemask_count);

  // Check if face is included in mask
  auto has_face = [](CellMask mask, Face face){ return mask & (1 << face); };
  // Add cells of block to facemask_* delimited by region (xmin:xmax, ...)
  auto add_cells = [&](CellMask mask, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
  {
    int dx = (xmax-xmin);
    int dy = (ymax-ymin);
    int dz = (zmax-zmin);

    int i0 = facemask_count_host(mask);
    facemask_count_host(mask) += dx*dy*dz;

    Kokkos::parallel_for( "add_cells", dx*dy*dz,
      KOKKOS_LAMBDA( int i )
    {
      int x = i%dx;
      int y = (i/dx)%dy;
      int z = (i/dx)/dy;
      x += xmin;
      y += ymin;
      z += zmin;

      uint32_t iCell = x + bx*y + bx*by*z;
      DYABLO_ASSERT_KOKKOS_DEBUG( i0 + i < facemask_iCells.extent(1), "precompute_facemask_cells : too many cells");
      facemask_iCells(mask, i0 + i) = iCell;
    });
  };

  int masks_count = (1 << Face::FACE_COUNT);
  DYABLO_ASSERT_HOST_RELEASE( ghost_count <= bx || ghost_count <= by, "GhostCommunicator_partial_blocks::init : ghost_count ("<<ghost_count<<") not compatible with block size (" << bx << "," << by << "," << bz << ")"  );

  for( CellMask mask = 1; mask < masks_count; mask++  )
  {

    // Uncomment to add full block each time for debug
    // add_cells( mask, 0, bx, 0, by, 0, bz );
    // continue;

    uint32_t xmin = 0, xmax = bx; // avoid duplicates : all cells with x<xmin are already added
    if( has_face(mask, Face::XL) )
    {
      int xl_xmax = ghost_count;
      add_cells( mask, 0, xl_xmax, 0, by, 0, bz );
      xmin = xl_xmax;
    }
    if( has_face(mask, Face::XR) )
    {
      int xr_xmin = std::max( bx-ghost_count, xmin ); // avoid adding cells already added when ghost_count < 2*bx
      add_cells( mask, xr_xmin, bx, 0, by, 0, bz );
      xmax = xr_xmin;
    }
    
    uint32_t ymin = 0, ymax = by;
    if( has_face(mask, Face::YL) )
    {
      int yl_ymax = ghost_count;
      add_cells( mask, xmin, xmax, 0, yl_ymax, 0, bz );
      ymin = yl_ymax;
    }
    if( has_face(mask, Face::YR) )
    {
      int yr_ymin = std::max( by-ghost_count, ymin );
      add_cells( mask, xmin, xmax, yr_ymin, by, 0, bz );
      ymax = yr_ymin;
    }

    uint32_t zmin = 0;
    if( has_face(mask, Face::ZL) )
    {
      int zl_zmax = ghost_count;
      add_cells( mask, xmin, xmax, ymin, ymax, 0, zl_zmax );
      zmin = zl_zmax;
    }
    if( has_face(mask, Face::ZR) )
    {
      int zr_zmin = std::max( (int)(bz-ghost_count), (int)zmin );
      add_cells( mask, xmin, xmax, ymin, ymax, zr_zmin, bz );
    }
  }

  Kokkos::deep_copy( facemask_count, facemask_count_host );


}


GhostCommunicator_partial_blocks::GhostCommunicator_partial_blocks( const AMRmesh_hashmap_new& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape, int ghost_count, const MpiComm& mpi_comm )
  : mpi_comm(mpi_comm)
{
  init(amr_mesh, shape, ghost_count, mpi_comm);
}

void GhostCommunicator_partial_blocks::init( const AMRmesh_hashmap_new& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape, int ghost_count, const MpiComm& mpi_comm )
{
  using GhostMap_t = AMRmesh_hashmap_new::GhostMap_t;

  int mpi_size = mpi_comm.MPI_Comm_size();

  DYABLO_ASSERT_HOST_RELEASE( ghost_count <= shape.bx || ghost_count <= shape.by, "GhostCommunicator_partial_blocks::init : ghost_count ("<<ghost_count<<") not compatible with block size (" << shape.bx << "," << shape.by << "," << shape.bz << ")"  );

  // Compute list of cells to send
  std::vector<int> send_sizes( mpi_size );
  Kokkos::View<uint32_t*> send_iOct;
  Kokkos::View<uint32_t*> send_iCell;
  {
    int masks_count = (1 << 6);
    int max_icells = shape.bx*shape.by*shape.bz;
    Kokkos::View<uint32_t*> facemask_count("facemask_count", masks_count);// number of cells to add for facemask
    Kokkos::View<uint32_t**> facemask_iCells("facemask_iCells", masks_count, max_icells);// cells to add for facemask
    precompute_facemask_cells( shape.bx, shape.by, shape.bz, ghost_count, facemask_count, facemask_iCells );

    GhostMap_t ghostmap = amr_mesh.getGhostMap();
    auto ghostmap_send_sizes_host = Kokkos::create_mirror_view(ghostmap.send_sizes);
    Kokkos::deep_copy( ghostmap_send_sizes_host, ghostmap.send_sizes );

    // Count cells to send to each process
    uint32_t first_rank_iOct = 0;
    for( int rank=0; rank<mpi_size; rank++ )
    {
      Kokkos::parallel_reduce("GhostCommunicator_partial_blocks::count_cells", ghostmap_send_sizes_host(rank),
        KOKKOS_LAMBDA( uint32_t iOct, int& count )
      {
        GhostMap_t::CellMask facemask = ghostmap.send_cell_masks(first_rank_iOct+iOct);
        count += facemask_count(facemask);

      }, send_sizes[rank]);
      first_rank_iOct += ghostmap_send_sizes_host(rank);
    }

    // Allocate cell containers
    uint32_t total_cell_count = std::accumulate( send_sizes.begin(), send_sizes.end(), 0 );
    send_iOct = Kokkos::View<uint32_t*>( "send_iOct", total_cell_count );
    send_iCell = Kokkos::View<uint32_t*>( "send_iCell", total_cell_count ); 

    // Fill cell containers
    Kokkos::parallel_scan("GhostCommunicator_partial_blocks::list_cells", ghostmap.send_iOcts.size(),
      KOKKOS_LAMBDA( uint32_t iOct, uint32_t& iCell_begin, bool final )
    {
      GhostMap_t::CellMask facemask = ghostmap.send_cell_masks(iOct);

      if( final )
      {
        for(uint32_t i=0; i<facemask_count(facemask); i++)
        {
          DYABLO_ASSERT_KOKKOS_DEBUG( (iCell_begin + i) < total_cell_count, "GhostCommunicator_partial_blocks::list_cells out of bounds " );
          send_iOct( iCell_begin + i ) = ghostmap.send_iOcts(iOct);
          send_iCell( iCell_begin + i ) = facemask_iCells( facemask, i);
        }
      }

      iCell_begin += facemask_count(facemask);
    });
  }

  // Get list of cells to recieve
  std::vector<int> recv_sizes( mpi_size );
  Kokkos::View< uint32_t* > recv_iOcts_local;
  Kokkos::View< uint32_t* > recv_iCell;
  int local_octant_count = 0;
  {
    // Exchange sizes
    mpi_comm.MPI_Alltoall( send_sizes.data(), 1, recv_sizes.data(), 1 );

    // Exchange iOct and iCell to pack/unpack
    uint32_t total_recv_count = std::accumulate( recv_sizes.begin(), recv_sizes.end(), 0);
    Kokkos::View< uint32_t* > recv_iOcts_remote( "recv_iOct_remote", total_recv_count );
    recv_iCell = Kokkos::View< uint32_t* >( "recv_iCell", total_recv_count );  
    Kokkos::fence();
    mpi_comm.MPI_Alltoallv( send_iOct.data(), send_sizes.data(), recv_iOcts_remote.data(), recv_sizes.data() );
    mpi_comm.MPI_Alltoallv( send_iCell.data(), send_sizes.data(), recv_iCell.data(), recv_sizes.data() );
    Kokkos::fence();
    
    // Convert iOct to local octants 
    recv_iOcts_local = Kokkos::View< uint32_t* >( "recv_iOcts", recv_iOcts_remote.layout() );
    uint32_t first_rank_ighost = 0;
    uint32_t first_rank_octant = 0;
    for( int rank=0; rank<mpi_size; rank++ )
    {
      uint32_t max_rank_octant = 0;
      Kokkos::parallel_scan( "convert_recv_iOct", (uint32_t)recv_sizes[rank],
        KOKKOS_LAMBDA( uint32_t ighost_local, uint32_t& current_rank_octant, bool final )
      {
        uint32_t ighost = first_rank_ighost + ighost_local;
        DYABLO_ASSERT_KOKKOS_DEBUG( ighost < recv_iOcts_local.size(), "convert_recv_iOct recv_iOcts_local out of bounds" );
        uint32_t current_local_octant = first_rank_octant + current_rank_octant;
        if( final )
        {
            recv_iOcts_local(ighost) = current_local_octant;
        }

        if( ighost==total_recv_count-1 || recv_iOcts_remote(ighost) != recv_iOcts_remote(ighost+1) ) 
        {
          current_rank_octant++;
        }
      }, max_rank_octant);
      first_rank_octant += max_rank_octant;
      first_rank_ighost += recv_sizes[rank];
    }

    local_octant_count = first_rank_octant;
  }

  this->m_send_cell_count = send_sizes;
  this->m_recv_cell_count = recv_sizes;
  this->m_send_iOct = send_iOct;
  this->m_send_iCell = send_iCell; 
  this->m_recv_iOct = recv_iOcts_local;
  this->m_recv_iCell = recv_iCell;
  this->m_local_ghost_octants = local_octant_count;
}

    
GhostCommunicator_partial_blocks::GhostCommunicator_partial_blocks( std::shared_ptr<AMRmesh> amr_mesh, const MpiComm& mpi_comm )
  : GhostCommunicator_partial_blocks(amr_mesh->getMesh(), mpi_comm)
{}

// /**
//  * TODO : doc
//  **/
// void GhostCommunicator_partial_blocks::reduce_ghosts( UserData::FieldAccessor& U) const
// {}

} // namespace dyablo