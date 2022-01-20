#pragma once

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "muscl_block/foreach_cell/ForeachCell.h"

namespace dyablo {
namespace muscl_block {

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
} //namespace muscl_block