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
                ForeachCell& foreach_cell,
                Timers& timers );
  ~GravitySolver_constant();
  void update_gravity_field( const ForeachCell::CellArray_global_ghosted& Uin,
                             const ForeachCell::CellArray_global_ghosted& Uout);

   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
} //namespace muscl_block