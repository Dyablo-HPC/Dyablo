#include "refine_condition/RefineCondition_helper.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "UserData.h"

namespace dyablo {


class RefineCondition_pseudo_gradient_formula
{
public:
  RefineCondition_pseudo_gradient_formula( ConfigMap& configMap )
    : error_min ( configMap.getValue<real_t>("amr", "epsilon_coarsen", 0.002) ), // TODO : pick better names
      error_max ( configMap.getValue<real_t>("amr", "epsilon_refine", 0.001) )
  {}

  enum VarIndex { ID };

  void init(const UserData& U)
  {
    this->Uin = U.getAccessor( {{"rho", ID}} );
  }

  template< int ndim >
  int getMarker( const ForeachCell::CellIndex& iCell, const ForeachCell::CellMetaData& cells ) const
  {
    using CellIndex = ForeachCell::CellIndex;

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
        else DYABLO_ASSERT_KOKKOS_DEBUG(false, "2:1 balance error : invalid level_diff");
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

    // -1 means coarsen
    //  0 means don't modify
    // +1 means refine
    int criterion = -1;
    if (diff_max > error_min)
      criterion = criterion < 0 ? 0 : criterion;
    if (diff_max > error_max)
      criterion = criterion < 1 ? 1 : criterion;

    return criterion;
  }

private:
  real_t error_min, error_max;
  UserData::FieldAccessor Uin;
};

/// Alias for template specialization 
class RefineCondition_pseudo_gradient 
  : public RefineCondition_helper<RefineCondition_pseudo_gradient_formula>
{
public:
   using RefineCondition_helper<RefineCondition_pseudo_gradient_formula>::RefineCondition_helper;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_pseudo_gradient, "RefineCondition_pseudo_gradient" );