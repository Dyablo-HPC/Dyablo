#pragma once

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"

namespace dyablo {
namespace muscl_block {

class MusclBlockUpdate{
public: 
  // MusclBlockUpdate(
  //               const ConfigMap& configMap,
  //               const HydroParams& params, 
  //               const LightOctree& lmesh,
  //               const id2index_t& fm,
  //               uint32_t bx, uint32_t by, uint32_t bz,
  //               Timers& timers );
  virtual ~MusclBlockUpdate(){}
  virtual void update(  DataArrayBlock U, DataArrayBlock Ughost,
                DataArrayBlock Uout, 
                real_t dt) = 0;
};

using MusclBlockUpdateFactory = RegisteringFactory< MusclBlockUpdate, 
  const ConfigMap& /*configMap*/,
  const HydroParams& /*params*/, 
  const LightOctree& /*lmesh*/,
  const id2index_t& /*fm*/,
  uint32_t /*bx*/, uint32_t /*by*/, uint32_t /*bz*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
} //namespace muscl_block