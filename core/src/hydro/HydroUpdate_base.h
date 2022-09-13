#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {


class HydroUpdate{
public: 
  // HydroUpdate(
  //               const ConfigMap& configMap,
  //               ForeachCell&& params, 
  //               const id2index_t& fm,
  //               uint32_t bx, uint32_t by, uint32_t bz,
  //               Timers& timers );
  virtual ~HydroUpdate(){}
  virtual void update(  const ForeachCell::CellArray_global_ghosted& Uin,
                        const ForeachCell::CellArray_global_ghosted& Uout,
                        real_t dt) = 0;
};

using HydroUpdateFactory = RegisteringFactory< HydroUpdate, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
