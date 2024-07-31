#include "refine_condition/RefineCondition_helper.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ParticleUpdate.h"

namespace dyablo {

/**
 * @brief Refines the mesh on downflows
 * 
 * We define a sign (downflow sign), a coarsening and a refining velocity.
 * The velocity chosen is the one of the last direction in the mesh (Y in 2D,
 * Z in 3D). Then we compute 
 * 
 */
class RefineCondition_downflows_formula
{
public:
  RefineCondition_downflows_formula( ConfigMap& configMap )
    : ndim( configMap.getValue<int>("mesh", "ndim") ),
      downflow_sign( configMap.getValue<real_t>("amr", "downflow_sign", 1.0)),
      velocity_coarsen( configMap.getValue<real_t>("amr", "velocity_coarsen", -0.1)),
      velocity_refine( configMap.getValue<real_t>("amr", "velocity_refine", 0.2))
  {}

  enum VarIndex { IRho, IRho_v };

  void init(const UserData& U, ScalarSimulationData& scalar_data)
  {
    this->Uin = U.getAccessor( {{"rho"                          , IRho  },
                                {ndim == 2 ? "rho_vy" : "rho_vz", IRho_v}} );
  }

  template< int ndim >
  KOKKOS_INLINE_FUNCTION
  int getMarker( const ForeachCell::CellIndex& iCell, const ForeachCell::CellMetaData& cells ) const
  {
    real_t rho = Uin.at(iCell, IRho);
    real_t v = downflow_sign * Uin.at(iCell, IRho_v)/rho;
    
    int criterion;
    if( v > velocity_refine )
      criterion = RefineCondition::REFINE;
    else if( v <= velocity_coarsen )
      criterion = RefineCondition::COARSEN;
    else
      criterion = RefineCondition::NOCHANGE;

    return criterion;
  }

private:
  int ndim;
  real_t downflow_sign;    // Sign of the velocity in downflow with respect to the orientation of the mesh
  real_t velocity_coarsen; // Velocity below which we coarsen the mesh
  real_t velocity_refine;  // Velocity above which we refine the mesh
  UserData::FieldAccessor Uin;
};

/// Alias for template specialization 
class RefineCondition_downflows 
  : public RefineCondition_helper<RefineCondition_downflows_formula>
{
public:
   using RefineCondition_helper<RefineCondition_downflows_formula>::RefineCondition_helper;
};

} // namespace dyablo 

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_downflows, "RefineCondition_downflows" );