#include "refine_condition/RefineCondition_helper.h"

#include "foreach_cell/ForeachCell.h"
#include "user_data/UserData.h"
#include "user_data/FieldAccessor.h"

namespace dyablo {


template< typename FieldPicker >
class RefineCondition_pseudo_gradient_formula
{
public:
  struct Params
  {
    Params( ConfigMap& configMap )
      : error_min ( configMap.getValue<real_t>("amr", "epsilon_coarsen", 0.002) ), // TODO : pick better names
        error_max ( configMap.getValue<real_t>("amr", "epsilon_refine", 0.001) )
    {}

    real_t error_min, error_max;
  };

  enum VarIndex { ID };

  RefineCondition_pseudo_gradient_formula(const Params& params, const UserData& U, ScalarSimulationData& scalar_data)
   : error_min( params.error_min ),
     error_max( params.error_max ),
     Uin( U.getAccessor( {{FieldPicker::getFieldName(), ID}} ) )
  {}

  template< int ndim >
  KOKKOS_INLINE_FUNCTION
  int getMarker( const ForeachCell::CellIndex& iCell, const ForeachCell::CellMetaData& cells ) const
  {
    using CellIndex = ForeachCell::CellIndex;

    ForeachCell::SearchMode_neighbor search_neighbor( cells.getLightOctree(), ForeachCell::SearchMode_neighbor::CLOSEST );

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
      max = FABS (qi - qj) / max;
      return FMAX (FMIN (max, 1.0), 0.0);        
    };

    auto grad = [&]( int dir, int sign )
    {
      CellIndex::offset_t offset{};
      offset[dir] = sign;
      CellIndex iCell_n = iCell.getNeighbor( offset, search_neighbor );

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
          for( int32_t dk=0; dk<dk_count; dk++ )
          for( int32_t dj=0; dj<dj_count; dj++ )
          for( int32_t di=0; di<di_count; di++ )
          {
              CellIndex iCell_n_smaller = iCell_n.getNeighbor({di,dj,dk}, search_neighbor); // This assumes that siblings are in same block
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

    int criterion;
    if( diff_max > error_max )
      criterion = RefineCondition::REFINE;
    else if( diff_max <= error_min )
      criterion = RefineCondition::COARSEN;
    else
      criterion = RefineCondition::NOCHANGE;

    return criterion;
  }

private:
  real_t error_min, error_max;
  UserData::FieldAccessor Uin;
};

/// Alias for template specialization 
struct FieldPicker_rho
{
  static std::string getFieldName()
  {
    return "rho";
  }
};
class RefineCondition_pseudo_gradient 
  : public RefineCondition_helper<RefineCondition_pseudo_gradient_formula<FieldPicker_rho>>
{
public:
   using RefineCondition_helper::RefineCondition_helper;
};

struct FieldPicker_xe
{
  static std::string getFieldName()
  {
    return "xe";
  }
};
class RefineCondition_pseudo_gradient_rad
  : public RefineCondition_helper<RefineCondition_pseudo_gradient_formula<FieldPicker_xe>>
{
public:
   using RefineCondition_helper::RefineCondition_helper;
};

/// Alias for template specialization 
struct FieldPicker_BZ
{
  static std::string getFieldName()
  {
    return "b";
  }
};
class RefineCondition_pseudo_gradient_BZ
  : public RefineCondition_helper<RefineCondition_pseudo_gradient_formula<FieldPicker_BZ>>
{
public:
   using RefineCondition_helper::RefineCondition_helper;
};

} // namespace dyablo 

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_pseudo_gradient, "RefineCondition_pseudo_gradient" );
FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_pseudo_gradient_rad, "RefineCondition_pseudo_gradient_rad" );
FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_pseudo_gradient_BZ, "RefineCondition_pseudo_gradient_BZ" );