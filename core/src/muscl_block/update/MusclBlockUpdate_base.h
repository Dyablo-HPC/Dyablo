#pragma once

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "muscl_block/foreach_cell/ForeachCell.h"

namespace dyablo {
namespace muscl_block {

class MusclBlockUpdate{
public: 
  // MusclBlockUpdate(
  //               const ConfigMap& configMap,
  //               ForeachCell&& params, 
  //               const id2index_t& fm,
  //               uint32_t bx, uint32_t by, uint32_t bz,
  //               Timers& timers );
  virtual ~MusclBlockUpdate(){}
  virtual void update(  const ForeachCell::CellArray_global_ghosted& Uin,
                        const ForeachCell::CellArray_global_ghosted& Uout,
                        real_t dt) = 0;
};

using MusclBlockUpdateFactory = RegisteringFactory< MusclBlockUpdate, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
} //namespace muscl_block