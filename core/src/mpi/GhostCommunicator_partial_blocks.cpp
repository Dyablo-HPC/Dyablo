#include "GhostCommunicator_partial_blocks.h"

namespace dyablo {

void precompute_facemask_cells( uint32_t bx, uint32_t by, uint32_t bz, uint32_t ghost_count, 
                                Kokkos::View<uint32_t*>& facemask_count, Kokkos::View<uint32_t**, Kokkos::LayoutRight>& facemask_iCells )
{
  using CellMask = AMRmesh_hashmap_new::GhostMap_t::CellMask;
  using Face = AMRmesh_hashmap_new::GhostMap_t::Face;

  auto facemask_count_host = Kokkos::create_mirror_view(facemask_count);

  // Check if face is included in mask
  auto has_face = [](CellMask mask, Face face){ return mask & (1 << face); };
  // Add cells of block to facemask_* delimited by region (xmin:xmax, ...)
  auto add_cells = [&](CellMask mask, uint32_t xmin, uint32_t xmax, uint32_t ymin, uint32_t ymax, uint32_t zmin, uint32_t zmax)
  {
    DYABLO_ASSERT_HOST_DEBUG( xmin <= xmax && xmax <= bx, "add_cells, xmin, xmax error, should have 0 <= xmin (" << xmin << ") <= xmax (" << xmax << ") <= bx (" << bx << ")" );
    DYABLO_ASSERT_HOST_DEBUG( ymin <= ymax && ymax <= by, "add_cells, ymin, ymax error, should have 0 <= ymin (" << ymin << ") <= ymax (" << ymax << ") <= by (" << by << ")" );
    DYABLO_ASSERT_HOST_DEBUG( zmin <= zmax && zmax <= bz, "add_cells, zmin, zmax error, should have 0 <= zmin (" << zmin << ") <= zmax (" << zmax << ") <= bz (" << bz << ")" );
    uint32_t dx = (xmax-xmin);
    uint32_t dy = (ymax-ymin);
    uint32_t dz = (zmax-zmin);

    uint32_t i0 = facemask_count_host(mask);
    facemask_count_host(mask) += dx*dy*dz;

    Kokkos::parallel_for( "add_cells", dx*dy*dz,
      KOKKOS_LAMBDA( uint32_t i )
    {
      uint32_t x = i%dx;
      uint32_t y = (i/dx)%dy;
      uint32_t z = (i/dx)/dy;
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
      uint32_t xl_xmax = ghost_count;
      add_cells( mask, 0, xl_xmax, 0, by, 0, bz );
      xmin = xl_xmax;
    }
    if( has_face(mask, Face::XR) )
    {
      uint32_t xr_xmin = std::max( bx-ghost_count, xmin ); // avoid adding cells already added when ghost_count < 2*bx
      add_cells( mask, xr_xmin, bx, 0, by, 0, bz );
      xmax = xr_xmin;
    }
    
    uint32_t ymin = 0, ymax = by;
    if( has_face(mask, Face::YL) )
    {
      uint32_t yl_ymax = ghost_count;
      add_cells( mask, xmin, xmax, 0, yl_ymax, 0, bz );
      ymin = yl_ymax;
    }
    if( has_face(mask, Face::YR) )
    {
      uint32_t yr_ymin = std::max( by-ghost_count, ymin );
      add_cells( mask, xmin, xmax, yr_ymin, by, 0, bz );
      ymax = yr_ymin;
    }

    uint32_t zmin = 0;
    if( has_face(mask, Face::ZL) )
    {
      uint32_t zl_zmax = std::min( ghost_count, bz );
      add_cells( mask, xmin, xmax, ymin, ymax, 0, zl_zmax );
      zmin = zl_zmax;
    }
    if( has_face(mask, Face::ZR) )
    {
      uint32_t zr_zmin = std::max( (int)bz-(int)ghost_count, (int)zmin);
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

  // Exchange ghostmap sizes
  GhostMap_t ghostmap = amr_mesh.getGhostMap();
  
  std::vector<int> ghostmap_send_sizes( mpi_size );
  {
    auto ghostmap_send_sizes_host = Kokkos::create_mirror_view(ghostmap.send_sizes);
    Kokkos::deep_copy( ghostmap_send_sizes_host, ghostmap.send_sizes );
    for( int i=0; i<mpi_size; i++ )
      ghostmap_send_sizes[i] = ghostmap_send_sizes_host(i);
  }
  std::vector<int> ghostmap_recv_sizes( mpi_size );
  mpi_comm.MPI_Alltoall(  ghostmap_send_sizes.data(), 1, 
                          ghostmap_recv_sizes.data(), 1);
  uint32_t total_ghostmap_recv_count = std::accumulate( ghostmap_recv_sizes.begin(), 
                                                        ghostmap_recv_sizes.end(), 0 );

  using CellMask = GhostMap_t::CellMask;
  // Send mask to recieving ranks
  Kokkos::View<CellMask*>& ghostmap_send_masks = ghostmap.send_cell_masks;
  Kokkos::View<CellMask*>  ghostmap_recv_masks("ghostmap_recv_masks", total_ghostmap_recv_count);

  #ifdef MPI_IS_CUDA_AWARE 
  {
    mpi_comm.MPI_Alltoallv( ghostmap_send_masks.data(), ghostmap_send_sizes.data(),
                            ghostmap_recv_masks.data(), ghostmap_recv_sizes.data());
  }
  #else
  {
    auto ghostmap_send_masks_host = Kokkos::create_mirror_view(ghostmap_send_masks);
    auto ghostmap_recv_masks_host = Kokkos::create_mirror_view(ghostmap_recv_masks);
    Kokkos::deep_copy(ghostmap_send_masks_host, ghostmap_send_masks);
    mpi_comm.MPI_Alltoallv( ghostmap_send_masks_host.data(), ghostmap_send_sizes.data(), 
                            ghostmap_recv_masks_host.data(), ghostmap_recv_sizes.data() );
    Kokkos::deep_copy(ghostmap_recv_masks, ghostmap_recv_masks_host);
  }
  #endif

  // Precompute cells for each mask type
  int masks_count = (1 << 6);
  int max_icells = shape.bx*shape.by*shape.bz;
  Kokkos::View<uint32_t*> facemask_count("facemask_count", masks_count);// number of cells to add for facemask
  Kokkos::View<uint32_t**, Kokkos::LayoutRight> facemask_iCells("facemask_iCells", masks_count, max_icells);// cells to add for facemask
  precompute_facemask_cells( shape.bx, shape.by, shape.bz, ghost_count, facemask_count, facemask_iCells );
  
  // Compute list of cells to send
  std::vector<int> send_sizes( mpi_size );
  Kokkos::View<uint32_t*> send_iOct;
  Kokkos::View<uint32_t*> send_iCell;
  {
    // Count cells to send to each process
    Kokkos::View<uint32_t*> offset_iOct("offset_iOct", ghostmap.send_cell_masks.size());
    uint32_t first_rank_iOct = 0;
    uint32_t offset = 0;
    for( int rank=0; rank<mpi_size; rank++ )
    {
      Kokkos::parallel_scan("GhostCommunicator_partial_blocks::count_send_cells", ghostmap_send_sizes[rank],
        KOKKOS_LAMBDA( uint32_t iOct, int& count, bool final )
      {
        GhostMap_t::CellMask facemask = ghostmap.send_cell_masks(first_rank_iOct+iOct);
        if( final )
          offset_iOct( first_rank_iOct+iOct ) = count + offset;
        count += facemask_count(facemask);
      }, send_sizes[rank]);
      first_rank_iOct += ghostmap_send_sizes[rank];
      offset+=send_sizes[rank];
    }

    // Allocate cell containers
    uint32_t total_cell_count = std::accumulate( send_sizes.begin(), send_sizes.end(), 0 );
    send_iOct = Kokkos::View<uint32_t*>( "send_iOct", total_cell_count );
    send_iCell = Kokkos::View<uint32_t*>( "send_iCell", total_cell_count ); 

    // Fill cell containers
    Kokkos::parallel_for("GhostCommunicator_partial_blocks::list_send_cells", 
      Kokkos::TeamPolicy<>(ghostmap.send_iOcts.size(), Kokkos::AUTO),
      KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type& team )
    {
      uint32_t iOct = team.league_rank();
      GhostMap_t::CellMask facemask = ghostmap.send_cell_masks(iOct);
      uint32_t iCell_begin = offset_iOct(iOct);
      uint32_t current_send_iOct = ghostmap.send_iOcts(iOct);

      Kokkos::parallel_for( Kokkos::TeamVectorRange(team, facemask_count(facemask)),
        [&]( uint32_t i )
      {
        DYABLO_ASSERT_KOKKOS_DEBUG( (iCell_begin + i) < total_cell_count, "GhostCommunicator_partial_blocks::list_cells out of bounds " );
        send_iOct( iCell_begin + i ) = current_send_iOct;
        send_iCell( iCell_begin + i ) = facemask_iCells( facemask, i);
      });
    });
  }


  // Compute list of cells to recieve
  std::vector<int> recv_sizes( mpi_size );
  Kokkos::View< uint32_t* > recv_iOcts;
  Kokkos::View< uint32_t* > recv_iCell;
  int local_octant_count = total_ghostmap_recv_count;
  {
    // Count cells to recieve from each process
    Kokkos::View<uint32_t*> offset_iOct("offset_iOct", ghostmap_recv_masks.size());
    uint32_t first_rank_iOct = 0;
    uint32_t offset = 0;
    for( int rank=0; rank<mpi_size; rank++ )
    {
      Kokkos::parallel_scan("GhostCommunicator_partial_blocks::count_recv_cells", ghostmap_recv_sizes[rank],
        KOKKOS_LAMBDA( uint32_t iOct, int& count, bool final )
      {
        GhostMap_t::CellMask facemask = ghostmap_recv_masks(first_rank_iOct+iOct);
        if( final )
          offset_iOct( first_rank_iOct+iOct ) = count + offset;
        count += facemask_count(facemask);

      }, recv_sizes[rank]);
      first_rank_iOct += ghostmap_recv_sizes[rank];
      offset+=recv_sizes[rank];
    }

    // Allocate cell containers
    uint32_t total_cell_count = std::accumulate( recv_sizes.begin(), recv_sizes.end(), 0 );
    recv_iOcts = Kokkos::View<uint32_t*>( "recv_iOct", total_cell_count );
    recv_iCell = Kokkos::View<uint32_t*>( "recv_iCell", total_cell_count ); 

    // Fill cell containers
    Kokkos::parallel_for("GhostCommunicator_partial_blocks::list_recv_cells", 
      Kokkos::TeamPolicy<>(local_octant_count, Kokkos::AUTO),
      KOKKOS_LAMBDA( const Kokkos::TeamPolicy<>::member_type& team )
    {
      uint32_t iOct = team.league_rank();
      GhostMap_t::CellMask facemask = ghostmap_recv_masks(iOct);
      uint32_t iCell_begin = offset_iOct(iOct);

      Kokkos::parallel_for( Kokkos::TeamVectorRange(team, facemask_count(facemask)),
        [&]( uint32_t i )
      {
        DYABLO_ASSERT_KOKKOS_DEBUG( (iCell_begin + i) < total_cell_count, "GhostCommunicator_partial_blocks::list_cells out of bounds " );
        recv_iOcts( iCell_begin + i ) = iOct;
        recv_iCell( iCell_begin + i ) = facemask_iCells( facemask, i);
      });
    });
  }

  this->m_send_cell_count = send_sizes;
  this->m_recv_cell_count = recv_sizes;
  this->m_send_iOct = send_iOct;
  this->m_send_iCell = send_iCell; 
  this->m_recv_iOct = recv_iOcts;
  this->m_recv_iCell = recv_iCell;
  this->m_local_ghost_octants = local_octant_count;
}

// /**
//  * TODO : doc
//  **/
// void GhostCommunicator_partial_blocks::reduce_ghosts( UserData::FieldAccessor& U) const
// {}

} // namespace dyablo