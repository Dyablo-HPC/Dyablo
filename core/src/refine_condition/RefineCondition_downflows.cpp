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
class RefineCondition_downflows : public RefineCondition
{
public:
  RefineCondition_downflows( ConfigMap& configMap,
                             ForeachCell& foreach_cell,
                             Timers& timers )
    : foreach_cell(foreach_cell),
      foreach_particle(foreach_cell.get_amr_mesh(), configMap),
      downflow_sign( configMap.getValue<real_t>("amr", "downflow_sign", 1.0)),
      velocity_coarsen( configMap.getValue<real_t>("amr", "velocity_coarsen", -0.1)),
      velocity_refine( configMap.getValue<real_t>("amr", "velocity_refine", 0.2))
  {}

  void mark_cells( UserData& Uin, ScalarSimulationData& scalar_data )
  {
    int ndim = foreach_cell.getDim();
    if (ndim == 2)
      mark_cells_aux<2>( Uin, scalar_data );
    else if (ndim == 3)
      mark_cells_aux<3>( Uin, scalar_data );
  }

  template< int ndim>
  void mark_cells_aux( UserData& Uin_, ScalarSimulationData& scalar_data ) 
  {
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    uint32_t nbOcts = foreach_cell.get_amr_mesh().getNumOctants();
    Kokkos::View<int*> oct_marker_max("Oct_marker_max", nbOcts);

    const real_t velocity_coarsen = this->velocity_coarsen; 
    const real_t velocity_refine  = this->velocity_refine;
    const real_t downflow_sign    = this->downflow_sign;

    // Putting density on the grid, for that we use NGP
    // What should we do when we have more than one particle array to refine on ?
    enum VarIndex_g{
      IRho, IRho_vy, IRho_vz
    };

    const UserData::FieldAccessor Uin = Uin_.getAccessor({{"rho", IRho},
                                                          {"rho_vy", IRho_vy},
                                                          {"rho_vz", IRho_vz}});

    // And marking cells
    foreach_cell.foreach_cell( "RefineCondition_mass::mark_cells", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      real_t rho = Uin.at(iCell, IRho);
      real_t v = (ndim == 2 ? downflow_sign * Uin.at(iCell, IRho_vy)/rho
                            : downflow_sign * Uin.at(iCell, IRho_vz)/rho);
      
      int criterion;
      if( v > velocity_refine )
        criterion = RefineCondition::REFINE;
      else if( v <= velocity_coarsen )
        criterion = RefineCondition::COARSEN;
      else
        criterion = RefineCondition::NOCHANGE;

      Kokkos::atomic_fetch_max( &oct_marker_max( iCell.getOct() ), criterion );
    });

    RefineCondition_utils::set_markers(foreach_cell.get_amr_mesh(), oct_marker_max);
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;

  real_t downflow_sign;    // Sign of the velocity in downflow with respect to the orientation of the mesh
  real_t velocity_coarsen; // Velocity below which we coarsen the mesh
  real_t velocity_refine;  // Velocity above which we refine the mesh
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_downflows, "RefineCondition_downflows" );