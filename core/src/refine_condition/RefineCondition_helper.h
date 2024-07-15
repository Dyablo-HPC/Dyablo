#pragma once

#include "refine_condition/RefineCondition_base.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "UserData.h"

namespace dyablo {

namespace RefineCondition_utils {

static void set_markers(AMRmesh& pmesh, const Kokkos::View<int*>& oct_marker)
{
  int level_min = pmesh.get_level_min();
  int level_max = pmesh.get_level_max();
  uint32_t nbOcts = pmesh.getNumOctants();
  const LightOctree& lmesh = pmesh.getLightOctree();    
  Kokkos::parallel_for( "MarkOctantsHydroFunctor::adjust_markers", nbOcts,
    KOKKOS_LAMBDA( uint32_t iOct )
  {
    uint8_t level = lmesh.getLevel({iOct,false});

    int criterion = oct_marker(iOct);

    // Don't coarsen/refine out of [level_min, level_max]
    if( level >= level_max && criterion==RefineCondition::REFINE )
      criterion = RefineCondition::NOCHANGE;
    if( level <= level_min && criterion==RefineCondition::COARSEN )
      criterion = RefineCondition::NOCHANGE;

    oct_marker( iOct ) = criterion;
  });

  pmesh.getMesh().setMarkers( oct_marker );
}

} // namespace RefineConditions_utils

template< typename RefineCondition_formula >
class RefineCondition_helper : public RefineCondition
{
public:
  RefineCondition_helper( ConfigMap& configMap,
                          ForeachCell& foreach_cell,
                          Timers& timers )
    : foreach_cell(foreach_cell),
      refineCondition_formula(configMap)
  {}

  void mark_cells( UserData& U, ScalarSimulationData& scalar_data)
  {
    int ndim = foreach_cell.getDim();
    if( ndim == 2 )
      mark_cells_aux<2>( U, scalar_data );
    else if( ndim == 3 )
      mark_cells_aux<3>( U, scalar_data );
  }

  template< int ndim >
  void mark_cells_aux( UserData& U, ScalarSimulationData& scalar_data )
  {
    using CellIndex = ForeachCell::CellIndex;

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    RefineCondition_formula& refineCondition_formula = this->refineCondition_formula;
    refineCondition_formula.init(U,scalar_data);

    uint32_t nbOcts = foreach_cell.get_amr_mesh().getNumOctants();
    Kokkos::View<int*> oct_marker_max("oct_marker_max", nbOcts);
    Kokkos::parallel_for( "fill_markers", oct_marker_max.size(),
      KOKKOS_LAMBDA( uint32_t iOct )
    {
      oct_marker_max(iOct) = -1;
    });
    foreach_cell.foreach_cell( "RefineCondition_helper::mark_cells", U.getShape(),
      KOKKOS_LAMBDA( const CellIndex& iCell )
    {
      int marker = refineCondition_formula.template getMarker<ndim>( iCell, cellmetadata );

      Kokkos::atomic_fetch_max( &oct_marker_max( iCell.getOct() ), marker );
    });

    RefineCondition_utils::set_markers(foreach_cell.get_amr_mesh(), oct_marker_max);
  }

private:
  ForeachCell& foreach_cell;
  RefineCondition_formula refineCondition_formula;
};

} // namespace dyablo 