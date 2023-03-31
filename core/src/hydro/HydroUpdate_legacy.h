#pragma once

#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "hydro/HydroUpdate_base.h"

class Timers;
class ConfigMap;

namespace dyablo {


class HydroUpdate_legacy : public HydroUpdate{
public: 
  HydroUpdate_legacy(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers  );
  ~HydroUpdate_legacy();
  void update( UserData& U, real_t dt) override;

   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
