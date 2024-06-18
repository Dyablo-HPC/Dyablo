#pragma once

#include "foreach_cell/ForeachCell_utils.h"
#include "HydroUpdate_utils.h"

namespace dyablo { 

/**
 * @brief Fills in block ghosts with neighboring data
 * 
 * This method is used when filling groups of blocks to copy the value of 
 * adjacent blocks. Here the "ghost" term refers to the cells surrounding a
 * a given block. They might be outside of the domain, or not. 
 * 
 * @param Uin[in] : Array to read the data from 
 * @param iCell_Ugroup[in] : Cell index in the frame of reference of the group
 * @param patch[in] : Cell metadata
 * @param bc_manager[in] : Boundary conditions to apply when filling cells outside of the domain
 * @param Ugroup[inout] : Array to update
 */
template< int ndim, typename State, typename Uin_t >
KOKKOS_INLINE_FUNCTION
void copyGhostBlockCellData(const Uin_t& Uin,
                            const CellIndex& iCell_Ugroup,
                            const ForeachCell::CellMetaData& patch,
                            const BoundaryConditions bc_manager, 
                            const PatchArray& Ugroup)
{
  using ConsState = typename State::ConsState;
  using CellIndex = ForeachCell::CellIndex;

  GhostedArray::Shape_t Uin_shape = Uin.getShape();

  CellIndex iCell_Uin = Uin_shape.convert_index_ghost(iCell_Ugroup);
  if( iCell_Uin.is_boundary() )
  {
    ConsState res = bc_manager.template getBoundaryValue<ndim, State>(Uin, iCell_Uin, patch);
    setConservativeState<ndim>(Ugroup, iCell_Ugroup, res);
  }
  else if( iCell_Uin.level_diff() >= 0 ) 
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( iCell_Uin.is_valid(), "Invalid iCell" );
    // Neighbor is bigger or same size : copy the only neighbor cell
    ConsState u;
    getConservativeState<ndim>( Uin, iCell_Uin, u );
    setConservativeState<ndim>( Ugroup, iCell_Ugroup, u );
  }
  else if( iCell_Uin.level_diff() == -1 ) 
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( iCell_Uin.is_valid(), "Invalid iCell" );
    ConsState u{}, u_subcell{};
    int nbCells =
    foreach_sibling<ndim>( iCell_Uin, Uin_shape, 
      [&](const CellIndex& iCell_subcell)
    {
      getConservativeState<ndim>(Uin, iCell_subcell, u_subcell);
      u += u_subcell;
    });
    setConservativeState<ndim>( Ugroup, iCell_Ugroup, u/nbCells );
  }
  else DYABLO_ASSERT_KOKKOS_DEBUG(false, "2:1 balance error : level_diff() is not -1, 0, 1");
}

}// namespace dyablo
