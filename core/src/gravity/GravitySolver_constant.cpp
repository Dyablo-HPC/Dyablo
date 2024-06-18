#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "gravity/GravitySolver_base.h"

class Timers;
namespace dyablo {

/**
 * @brief Class applying a constant gravity acceleration to the grid.
*/
class GravitySolver_constant : public GravitySolver{
public: 
  GravitySolver_constant(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
  { 
    [[maybe_unused]] GravityType gtype = configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_CST_SCALAR);
    DYABLO_ASSERT_HOST_RELEASE( gtype == GRAVITY_CST_SCALAR, "GravitySolver_cg must have gravity_type=constant_scalar" );
  }
  ~GravitySolver_constant(){}
  void update_gravity_field( UserData& U, ScalarSimulationData& scalar_data )
  {
    /* Does nothing : constant values are used when applying gravity */
  }
};

} //namespace dyablo 

FACTORY_REGISTER( dyablo::GravitySolverFactory, dyablo::GravitySolver_constant, "GravitySolver_constant" );

