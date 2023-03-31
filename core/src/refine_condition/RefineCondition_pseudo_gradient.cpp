#include "refine_condition/RefineCondition_base.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "UserData.h"

namespace dyablo {


class RefineCondition_pseudo_gradient : public RefineCondition
{
public:
  RefineCondition_pseudo_gradient( ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : foreach_cell(foreach_cell),
      //timers(timers),
      error_min ( configMap.getValue<real_t>("amr", "epsilon_coarsen", 0.002) ), // TODO : pick better names
      error_max ( configMap.getValue<real_t>("amr", "epsilon_refine", 0.001) )
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

    real_t error_min = this->error_min;
    real_t error_max = this->error_max;

    enum VarIndex_rho{ID};

    UserData::FieldAccessor Uin( U, {{"rho", ID}} );
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    uint32_t nbOcts = foreach_cell.get_amr_mesh().getNumOctants();
    Kokkos::View<real_t*> oct_err_max("Oct_err_max", nbOcts);
    foreach_cell.foreach_cell( "RefineCondition_generic::mark_cells", Uin.getShape(),
      KOKKOS_LAMBDA( const CellIndex& iCell )
    {

      /**
       * Indicator - scalar gradient.
       *
       * Adapted from CanoP for comparison.
       * returned value is between 0 and 1.
       * - a small value probably means no refinement necessary
       * - a high value probably means refinement should be activated
       */
      auto indicator_scalar_gradient = [&](real_t qi, real_t qj) -> real_t
      {
        real_t max = FMAX (FABS (qi), FABS (qj));        
        if (max < 0.001) {
          return 0;
        }        
        max = FABS (qi - qj) / max;
        return FMAX (FMIN (max, 1.0), 0.0);        
      };

      auto grad = [&]( int dir, int sign )
      {
        CellIndex::offset_t offset{};
        offset[dir] = sign;
        CellIndex iCell_n = iCell.getNeighbor_ghost( offset, Uin.getShape() );

        real_t diff_max = 0;

        if( !iCell_n.is_boundary() )
        {
          if( iCell_n.level_diff() >=0 )
          {
            diff_max = indicator_scalar_gradient( Uin.at(iCell, ID), Uin.at(iCell_n, ID)); 
          }
          else if( iCell_n.level_diff() == -1 )
          {
            // Neighbor is smaller : get max diff with siblings
            // Iterate over adjacent neighbors
            int di_count = (offset[IX]==0)?2:1;
            int dj_count = (offset[IY]==0)?2:1;
            int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
            for( int8_t dk=0; dk<dk_count; dk++ )
            for( int8_t dj=0; dj<dj_count; dj++ )
            for( int8_t di=0; di<di_count; di++ )
            {
                CellIndex iCell_n_smaller = iCell_n.getNeighbor_ghost({di,dj,dk}, Uin.getShape()); // This assumes that siblings are in same block
                real_t diff = indicator_scalar_gradient( Uin.at(iCell, ID), Uin.at(iCell_n_smaller, ID)); 
                diff_max = FMAX( diff_max, diff );
            }

          }
          else assert(false); // Should not happen
        }
        return diff_max;
      };

      real_t diff_max = 0;
      diff_max = FMAX( diff_max, grad(IX, +1) );
      diff_max = FMAX( diff_max, grad(IX, -1) );
      diff_max = FMAX( diff_max, grad(IY, +1) );
      diff_max = FMAX( diff_max, grad(IY, -1) );
      if(ndim==3)
      {
        diff_max = FMAX( diff_max, grad(IZ, +1) );
        diff_max = FMAX( diff_max, grad(IZ, -1) );
      }

      Kokkos::atomic_fetch_max( &oct_err_max( iCell.getOct() ), diff_max );
    });

    AMRmesh& pmesh = foreach_cell.get_amr_mesh();
    int level_min = pmesh.get_level_min();
    int level_max = pmesh.get_level_max();
    const LightOctree& lmesh = pmesh.getLightOctree();    
    Kokkos::View<uint32_t*> markers_iOct("markers_iOct", nbOcts);
    Kokkos::View<int*> markers_marker("markers_marker", nbOcts);
    uint32_t nb_markers = 0;
    Kokkos::parallel_scan( "MarkOctantsHydroFunctor::compress_markers", nbOcts,
      KOKKOS_LAMBDA( uint32_t iOct, uint32_t& nb_markers, bool final )
    {
      uint8_t level = lmesh.getLevel({iOct,false});
      real_t error = oct_err_max(iOct);

      // -1 means coarsen
      //  0 means don't modify
      // +1 means refine
      int criterion = -1;
      if (error > error_min)
        criterion = criterion < 0 ? 0 : criterion;
      if (error > error_max)
        criterion = criterion < 1 ? 1 : criterion;

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

private:
  ForeachCell& foreach_cell;
  //Timers& timers;
  real_t error_min, error_max;  
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_pseudo_gradient, "RefineCondition_pseudo_gradient" );