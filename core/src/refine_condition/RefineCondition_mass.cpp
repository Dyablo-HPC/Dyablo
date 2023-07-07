#include "refine_condition/RefineCondition_helper.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ParticleUpdate.h"


namespace dyablo {

namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray = ForeachCell::CellArray_global;
using PatchArray = ForeachCell::CellArray_patch;
using CellIndex = ForeachCell::CellIndex;

}// namespace
}// namespace dyablo


#include "hydro/CopyGhostBlockCellData.h"

namespace dyablo {

class RefineCondition_mass : public RefineCondition
{
public:
  RefineCondition_mass( ConfigMap& configMap,
                        ForeachCell& foreach_cell,
                        Timers& timers )
    : foreach_cell(foreach_cell),
      foreach_particle(foreach_cell.get_amr_mesh(), configMap),
      mass_coarsen( configMap.getValue<real_t>("amr", "mass_coarsen", 0.1)),
      mass_refine( configMap.getValue<real_t>("amr", "mass_refine", 1.0))
  {
    std::string particle_update_density_id = configMap.getValue<std::string>("particles", "update_density", "none");
    particle_update_density = ParticleUpdateFactory::make_instance( particle_update_density_id,
      configMap,
      foreach_cell,
      timers
    );
  }

  void mark_cells( UserData& Uin, ScalarSimulationData& scalar_data )
  {
    int ndim = foreach_cell.getDim();
    if( ndim == 2 )
      mark_cells_aux<2, PrimHydroState, ConsHydroState>( Uin, scalar_data );
    else if( ndim == 3 )
      mark_cells_aux<3, PrimHydroState, ConsHydroState>( Uin, scalar_data );
  }

  template< int ndim,
            typename PrimState,
            typename ConsState >
  void mark_cells_aux( UserData& Uin_, ScalarSimulationData& scalar_data ) 
  {
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    uint32_t nbOcts = foreach_cell.get_amr_mesh().getNumOctants();
    Kokkos::View<int*> oct_marker_max("Oct_marker_max", nbOcts);

    const real_t dt = scalar_data.get<real_t>("dt");

    const real_t mass_coarsen = this->mass_coarsen; 
    const real_t mass_refine  = this->mass_refine;

    std::string field_name = "rho";
    if (particle_update_density) {
      if (!Uin_.has_field("rho_g"))
        Uin_.new_fields({"rho_g"});
      particle_update_density->update( Uin_, dt );
      field_name = "rho_g";
    }

    // Putting density on the grid, for that we use NGP
    // What should we do when we have more than one particle array to refine on ?
    enum VarIndex_g{
      IRho
    };

    const UserData::FieldAccessor Uin = Uin_.getAccessor({{field_name, IRho} });

    // And marking cells
    foreach_cell.foreach_cell( "RefineCondition_mass::mark_cells", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto size = cellmetadata.getCellSize(iCell);
      real_t local_mass = Uin.at(iCell, IRho) * size[IX]*size[IY]*(ndim == 3 ? size[IZ] : 1.0);

      
      int criterion;
      if( local_mass > mass_refine )
        criterion = RefineCondition::REFINE;
      else if( local_mass <= mass_coarsen )
        criterion = RefineCondition::COARSEN;
      else
        criterion = RefineCondition::NOCHANGE;

      Kokkos::atomic_fetch_max( &oct_marker_max( iCell.getOct() ), criterion );
    });

    if (particle_update_density)
      Uin_.delete_field({"rho_g"});

    RefineCondition_utils::set_markers(foreach_cell.get_amr_mesh(), oct_marker_max);
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;

  real_t mass_coarsen, mass_refine;

  std::unique_ptr<ParticleUpdate> particle_update_density;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_mass, "RefineCondition_mass" );