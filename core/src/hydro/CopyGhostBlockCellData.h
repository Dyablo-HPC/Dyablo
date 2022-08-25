#pragma once

#include "foreach_cell/ForeachCell_utils.h"
#include "HydroUpdate_utils.h"

namespace dyablo { 

template< int ndim, typename State >
KOKKOS_INLINE_FUNCTION
void copyGhostBlockCellData(const GhostedArray& Uin, const CellIndex& iCell_Ugroup,
                            const ForeachCell::CellMetaData& patch,
                            const BoundaryConditions<State> bc_manager, 
                            const PatchArray& Ugroup)
{
  using ConsState = typename State::ConsState;
  using CellIndex = ForeachCell::CellIndex;

  CellIndex iCell_Uin = Uin.convert_index_ghost(iCell_Ugroup);
  if( iCell_Uin.is_boundary() )
  {
    ConsState res = bc_manager.template getBoundaryValue<ndim>(Uin, iCell_Uin, patch);
    setConservativeState<ndim>(Ugroup, iCell_Ugroup, res);
  }
  else if( iCell_Uin.level_diff() >= 0 ) 
  {
    assert( iCell_Uin.is_valid() );
    // Neighbor is bigger or same size : copy the only neighbor cell
    ConsState u;
    getConservativeState<ndim>( Uin, iCell_Uin, u );
    setConservativeState<ndim>( Ugroup, iCell_Ugroup, u );
  }
  else if( iCell_Uin.level_diff() == -1 ) 
  {
    assert( iCell_Uin.is_valid() );
    ConsState u{}, u_subcell{};
    int nbCells =
    foreach_sibling<ndim>( iCell_Uin, Uin, 
      [&](const CellIndex& iCell_subcell)
    {
      getConservativeState<ndim>(Uin, iCell_subcell, u_subcell);
      u += u_subcell;
    });
    setConservativeState<ndim>( Ugroup, iCell_Ugroup, u/nbCells );
  }
  else assert(false); // Should not happen
}

}// namespace dyablo
