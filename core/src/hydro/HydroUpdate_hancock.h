#pragma once

#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "hydro/HydroUpdate_base.h"

class Timers;
class ConfigMap;

namespace dyablo {


class HydroUpdate_hancock : public HydroUpdate{
public: 
  HydroUpdate_hancock(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers );
  ~HydroUpdate_hancock();
  void update(  const ForeachCell::CellArray_global_ghosted& Uin,
                const ForeachCell::CellArray_global_ghosted& Uout,
                real_t dt);
   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
