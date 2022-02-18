#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {


class GravitySolver{
public: 
  // GravitySolver(
  //               const ConfigMap& configMap,
  //               ForeachCell& foreach_cell,
  //               Timers& timers );
  virtual ~GravitySolver(){}
  virtual void update_gravity_field( const ForeachCell::CellArray_global_ghosted& Uin,
                                     const ForeachCell::CellArray_global_ghosted& Uout) = 0;
};

using GravitySolverFactory = RegisteringFactory< GravitySolver, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
