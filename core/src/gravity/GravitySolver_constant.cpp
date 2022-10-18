#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "gravity/GravitySolver_base.h"

class Timers;
namespace dyablo {


class GravitySolver_constant : public GravitySolver{
public: 
  GravitySolver_constant(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
  { 
    GravityType gtype = configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_CST_SCALAR);
    if(gtype != GRAVITY_CST_SCALAR)
      throw std::runtime_error("GravitySolver_constant must have gravity_type=constant_scalar");
  }
  ~GravitySolver_constant(){}
  void update_gravity_field( const ForeachCell::CellArray_global_ghosted& Uin,
                             const ForeachCell::CellArray_global_ghosted& Uout)
  {
    /* Does nothing : constant values are used when applying gravity */
  }
};

} //namespace dyablo 

FACTORY_REGISTER( dyablo::GravitySolverFactory, dyablo::GravitySolver_constant, "GravitySolver_constant" );

