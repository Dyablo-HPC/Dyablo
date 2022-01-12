#pragma once

#include <memory>

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "muscl_block/gravity/GravitySolver_base.h"

class Timers;
class ConfigMap;

namespace dyablo {
namespace muscl_block {

class GravitySolver_constant : public GravitySolver{
public: 
  GravitySolver_constant(
                ConfigMap& configMap,
                std::shared_ptr<AMRmesh> pmesh,
                const id2index_t& fm,
                uint32_t bx, uint32_t by, uint32_t bz,
                Timers& timers );
  ~GravitySolver_constant();
  void update_gravity_field(  DataArrayBlock U, DataArrayBlock Ughost,
                DataArrayBlock Uout);

   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
} //namespace muscl_block