#pragma once

#include "kokkos_shared.h"

namespace dyablo {

namespace{

using CellIndex = ForeachCell::CellIndex;
using CellArray_global_ghosted = ForeachCell::CellArray_global_ghosted;

template< bool enable_different_block >
KOKKOS_INLINE_FUNCTION
CellIndex get_sibling_cell(const CellIndex& iCell, const CellIndex::offset_t& offset, const CellArray_global_ghosted& array);

template<>
KOKKOS_INLINE_FUNCTION
CellIndex get_sibling_cell<false>(const CellIndex& iCell, const CellIndex::offset_t& offset, const CellArray_global_ghosted& /*array*/)
{
  return iCell.getNeighbor(offset);
}

template<>
KOKKOS_INLINE_FUNCTION
CellIndex get_sibling_cell<true>(const CellIndex& iCell, const CellIndex::offset_t& offset, const CellArray_global_ghosted& array)
{
  return iCell.getNeighbor_ghost(offset, array);
}

}

/**
 * Iterate over neighbors when current cell is a smaller neighbor
 * @tparam ndim 2D/3D
 * @tparam enable_different_block specify if all smaller neighbors are guaranteed to be in the same block
 *          if true (i.g. cell-based or odd block size), getNeighbor_ghost() is used to find siblings
 *          if false, getNeighbor() is used because no tree search is needed
 * @param offset is the offset that was applied to get current CellIndex
 * @param apply_neighbor is a (const CellIndex&) -> void functor that performs an operation with 
 *                       each neighbor smaller cell
 * @param array the array that needs to be accessed (mesh used for neighbor search in AMR tree)
 * @returns number of sibling cells
 * NOTE : level_diff() must be -1
 * NOTE : for example, neighbors in 3D are the 4 cells that are in contact with the original cell
 **/
template< int ndim, bool enable_different_block=true, typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_smaller_neighbor( const CellIndex& iCell, const CellIndex::offset_t& offset, const CellArray_global_ghosted& array, const Func& apply_neighbor )
{
  assert( iCell.level_diff() == -1 );
  // enable_different_block must be activated for cell-based or odd block size
  assert( enable_different_block || ( iCell.bx%2 == 0 && iCell.by%2 == 0 && iCell.bz%2 == 0 ) );

  int di_count = (offset[IX]==0)?2:1;
  int dj_count = (offset[IY]==0)?2:1;
  int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
  for( int8_t dk=0; dk<dk_count; dk++ )
  for( int8_t dj=0; dj<dj_count; dj++ )
  for( int8_t di=0; di<di_count; di++ )
  {
      CellIndex iCell_ghost = get_sibling_cell<enable_different_block>( iCell, {di,dj,dk}, array );
      apply_neighbor(iCell_ghost);
  }
  return di_count*dj_count*dk_count;
}

/**
 * Iterate over sibling cells
 * @param apply_neighbor is a (const CellIndex&) -> void functor that performs an operation with 
 *                       each sibling cell
 * @param iCell first cell from the bigger supercell (sibling with the smallest morton index)
 * @param apply_neighbor is a (const CellIndex&) -> void functor that performs an operation with 
 *                       each sibling
 * @returns number of sibling cells
 * NOTE : for example in 3D, sibings are the 8 cells that form a bigger supercell
 **/
template< int ndim, bool enable_different_block=true, typename Func >
KOKKOS_INLINE_FUNCTION
int foreach_sibling( const CellIndex& iCell, const CellArray_global_ghosted& array, const Func& apply_sibling )
{
  // enable_different_block must be activated for cell-based or odd block size
  assert( enable_different_block || ( iCell.bx%2 == 0 && iCell.by%2 == 0 && iCell.bz%2 == 0 ) );

  int dk_count = ndim==3?2:1;
  for( int8_t dk=0; dk<dk_count; dk++ )
  for( int8_t dj=0; dj<2; dj++ )
  for( int8_t di=0; di<2; di++ )
  {
      CellIndex iCell_ghost = get_sibling_cell<enable_different_block>( iCell, {di,dj,dk}, array );
      apply_sibling(iCell_ghost);
  }
  return 2*2*dk_count;
}

} // namespace dyablo