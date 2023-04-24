#pragma once

#include "amr/LightOctree.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo{

/**
 * Helper class to find mapping between and old amr mesh and a newly refined one
 **/
class CellIndexRemapper{
public:
  using CellIndex = AMRBlockForeachCell_CellArray_impl::CellIndex;
  using OctantIndex = LightOctree::OctantIndex;
  CellIndexRemapper( const LightOctree& lmesh_old, ForeachCell& foreach_cell_new )
    : lmesh_old( lmesh_old ),
      lmesh_new( foreach_cell_new.get_amr_mesh().getLightOctree() )
  {}

  /**
   * Convert cell index from the current amr mesh to an index pointing to 
   * the corresponding cell in the old amr mesh
   * 
   * @param iCell_new A CellIndex in the current amr mesh
   **/
  KOKKOS_INLINE_FUNCTION
  CellIndex get_old_cell( const CellIndex& iCell_new ) const
  {
    int ndim = lmesh_old.getNdim();

    DYABLO_ASSERT_KOKKOS_DEBUG( iCell_new.is_valid(), "Invalid cell" );
    DYABLO_ASSERT_KOKKOS_DEBUG( !iCell_new.iOct.isGhost, "Cell can't be ghost" );

    LightOctree::pos_t oct_pos_new = lmesh_new.getCenter(iCell_new.iOct);
    auto oct_size_new = lmesh_new.getSize(iCell_new.iOct);
    LightOctree::pos_t first_old_pos = {
      oct_pos_new[IX] - 0.25*oct_size_new[IX],
      oct_pos_new[IY] - 0.25*oct_size_new[IY],
      (ndim==2) ? 0 : oct_pos_new[IZ] - 0.25*oct_size_new[IZ]
    };
    OctantIndex iOct_old = lmesh_old.getiOctFromPos( first_old_pos );

    int level_old = lmesh_old.getLevel( iOct_old );
    int level_new = lmesh_new.getLevel( iCell_new.iOct );;

    if( level_old == level_new ) // same size, same cell in block
    {
      CellIndex iCell_old = iCell_new;
      iCell_old.iOct = iOct_old;
      return iCell_old;
    }
    else if( level_old == level_new-1 ) // old cell was bigger
    {
      // Compute position of new octant in bigger source octant
      auto oct_pos_old = lmesh_old.getCenter(iOct_old);
      // new (smaller) suboctant is right suboctant if it's center is right of old (bigger) octant 
      int oct_offset_x = oct_pos_new[IX] > oct_pos_old[IX];
      int oct_offset_y = oct_pos_new[IY] > oct_pos_old[IY];
      int oct_offset_z = oct_pos_new[IZ] > oct_pos_old[IZ];
      // Compute position inside old (bigger) octant
      uint32_t i_bigger = (iCell_new.i + iCell_new.bx * oct_offset_x) / 2;
      uint32_t j_bigger = (iCell_new.j + iCell_new.by * oct_offset_y) / 2;
      uint32_t k_bigger = (iCell_new.k + iCell_new.bz * oct_offset_z) / 2;

      CellIndex iCell_old{
        iOct_old,
        i_bigger, j_bigger, k_bigger,
        iCell_new.bx, iCell_new.by, iCell_new.bz, 
        CellIndex::Status::BIGGER
      };

      return iCell_old;
    }
    else if( level_old == level_new+1 ) // old cells were smaller
    {
      // Compute suboctant offset 
      // Use >(bx-1)/2 to take into account odd bx
      int8_t i_oct_offset = (iCell_new.i > (iCell_new.bx-1)/2);
      int8_t j_oct_offset = (iCell_new.j > (iCell_new.by-1)/2);
      int8_t k_oct_offset = (iCell_new.k > (iCell_new.bz-1)/2);
      // Get Suboctant index 
      auto oct_neighbors = lmesh_old.findNeighbors( iOct_old, {i_oct_offset, j_oct_offset, k_oct_offset} );
      DYABLO_ASSERT_KOKKOS_DEBUG( oct_neighbors.size() == 1, "Siblings should be same size when coarsening" );
      OctantIndex iOct_old_smaller = oct_neighbors[0];

      // Compute position in suboctant
      uint32_t i_smaller = iCell_new.i*2 - i_oct_offset*iCell_new.bx;
      uint32_t j_smaller = iCell_new.j*2 - j_oct_offset*iCell_new.by;
      uint32_t k_smaller = iCell_new.k*2 - k_oct_offset*iCell_new.bz;      

      CellIndex iCell_old{
        iOct_old_smaller,
        i_smaller, j_smaller, k_smaller,
        iCell_new.bx, iCell_new.by, iCell_new.bz, 
        CellIndex::Status::SMALLER
      };

      return iCell_old;
    }
    else
    {
      DYABLO_ASSERT_KOKKOS_DEBUG(false, "2:1 infringement during remap");
      return CELLINDEX_INVALID;
    }   
  }
private:
  LightOctree lmesh_old, lmesh_new;
};

} // namespace dyablo