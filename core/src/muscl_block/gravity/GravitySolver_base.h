#pragma once

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"

namespace dyablo {
namespace muscl_block {

class GravitySolver{
public: 
  // GravitySolver(
  //               const ConfigMap& configMap,
  //               shared_ptr<AMRmesh> pmesh,
  //               const id2index_t& fm,
  //               uint32_t bx, uint32_t by, uint32_t bz,
  //               Timers& timers );
  virtual ~GravitySolver(){}
  virtual void update_gravity_field(  DataArrayBlock U, DataArrayBlock Ughost,
                DataArrayBlock Uout) = 0;
};

using GravitySolverFactory = RegisteringFactory< GravitySolver, 
  ConfigMap& /*configMap*/,
  std::shared_ptr<AMRmesh> /*pmesh*/,
  const id2index_t& /*fm*/,
  uint32_t /*bx*/, uint32_t /*by*/, uint32_t /*bz*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
} //namespace muscl_block