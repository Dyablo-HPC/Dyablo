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
  Kokkos::View<uint32_t*> markers_iOct("markers_iOct", nbOcts);
  Kokkos::View<int*> markers_marker("markers_marker", nbOcts);
  uint32_t nb_markers = 0;
  Kokkos::parallel_scan( "MarkOctantsHydroFunctor::compress_markers", nbOcts,
    KOKKOS_LAMBDA( uint32_t iOct, uint32_t& nb_markers, bool final )
  {
    uint8_t level = lmesh.getLevel({iOct,false});

    // -1 means coarsen
    //  0 means don't modify
    // +1 means refine
    int criterion = oct_marker(iOct);

    // Don't coarsen/refine out of [level_min, level_max]
    if( level >= level_max && criterion==1 )
      criterion = 0;
    if( level <= level_min && criterion==-1 )
      criterion = 0;

    if( criterion != 0 )
    {
      if( final )
      {
        markers_iOct(nb_markers) = iOct;
        markers_marker(nb_markers) = criterion;
      }
      nb_markers ++;
    }
  }, nb_markers);

  auto markers_iOct_host = Kokkos::create_mirror_view(markers_iOct);
  auto markers_marker_host = Kokkos::create_mirror_view(markers_marker);
  Kokkos::deep_copy( markers_iOct_host, markers_iOct );
  Kokkos::deep_copy( markers_marker_host, markers_marker );

  Kokkos::parallel_for( "MarkOctantsHydroFunctor::set_markers_pablo", 
                      Kokkos::RangePolicy<Kokkos::OpenMP>(0,nb_markers),
                      [&](uint32_t i)
  {
    uint32_t iOct = markers_iOct_host(i);
    int marker = markers_marker_host(i);

    pmesh.setMarker(iOct, marker);
  });
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

  void mark_cells( const UserData& U )
  {
    int ndim = foreach_cell.getDim();
    if( ndim == 2 )
      mark_cells_aux<2>( U );
    else if( ndim == 3 )
      mark_cells_aux<3>( U );
  }

  template< int ndim >
  void mark_cells_aux( const UserData& U )
  {
    using CellIndex = ForeachCell::CellIndex;

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    RefineCondition_formula& refineCondition_formula = this->refineCondition_formula;
    refineCondition_formula.init(U);

    uint32_t nbOcts = foreach_cell.get_amr_mesh().getNumOctants();
    Kokkos::View<int*> oct_marker_max("oct_marker_max", nbOcts);
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