#pragma once

#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "gravity/GravitySolver_base.h"

class Timers;
class ConfigMap;

namespace dyablo {


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
