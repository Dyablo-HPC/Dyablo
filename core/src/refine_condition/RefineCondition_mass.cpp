#include "refine_condition/RefineCondition_helper.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ParticleUpdate.h"

namespace dyablo {

class RefineCondition_mass_formula
{
public:
  struct Params
  {
    Params( ConfigMap& configMap )
    : mass_coarsen( configMap.getValue<real_t>("amr", "mass_coarsen", 0.1)),
      mass_refine( configMap.getValue<real_t>("amr", "mass_refine", 1.0))
    {}

    real_t mass_coarsen, mass_refine;
    std::string rho_field_name = "rho";
  };

  enum VarIndex { Irho };

  RefineCondition_mass_formula( const Params& params, const UserData& U, ScalarSimulationData& scalar_data )
    : mass_coarsen( params.mass_coarsen ),
      mass_refine( params.mass_refine ),
      Uin( U.getAccessor( {{params.rho_field_name, Irho  }} ))
  {}

  template< int ndim >
  KOKKOS_INLINE_FUNCTION
  int getMarker( const ForeachCell::CellIndex& iCell, const ForeachCell::CellMetaData& cells ) const
  {
    auto size = cells.getCellSize(iCell);
    real_t local_mass = Uin.at(iCell, Irho) * size[IX]*size[IY]*(ndim == 3 ? size[IZ] : 1.0);
    
    int criterion;
    if( local_mass > mass_refine )
      criterion = RefineCondition::REFINE;
    else if( local_mass <= mass_coarsen )
      criterion = RefineCondition::COARSEN;
    else
      criterion = RefineCondition::NOCHANGE;

    return criterion;
  }

  real_t mass_coarsen, mass_refine;
  UserData::FieldAccessor Uin;
};

/// Alias for template specialization 
class RefineCondition_mass
  : public RefineCondition_helper<RefineCondition_mass_formula>
{
protected:
  std::unique_ptr<ParticleUpdate> particle_update_density;

public:
  RefineCondition_mass( ConfigMap& configMap,
                        ForeachCell& foreach_cell,
                        Timers& timers )
  : RefineCondition_helper( configMap, foreach_cell, timers )
  {
    std::string particle_update_density_id = configMap.getValue<std::string>("particles", "update_density", "none");
    particle_update_density = ParticleUpdateFactory::make_instance( particle_update_density_id,
      configMap,
      foreach_cell,
      timers
    );
  }

  void mark_cells( UserData& U, ScalarSimulationData& scalar_data )
  {
    std::string field_name = "rho";
    if (particle_update_density) {
      if (!U.has_field("rho_g"))
        U.new_fields({"rho_g"});
      particle_update_density->update( U, scalar_data );
      field_name = "rho_g";
    }

    refineCondition_formula_params.rho_field_name = field_name;

    RefineCondition_helper::mark_cells(U, scalar_data);

    if (particle_update_density)
      U.delete_field( field_name );
  }

};

} // namespace dyablo 

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_mass, "RefineCondition_mass" );