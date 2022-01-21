#pragma once

#include <memory>

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "muscl_block/update/MusclBlockUpdate_base.h"

class Timers;
class ConfigMap;

namespace dyablo {
namespace muscl_block {

class MusclBlockUpdate_legacy : public MusclBlockUpdate{
public: 
  MusclBlockUpdate_legacy(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers  );
  ~MusclBlockUpdate_legacy();
  void update(  const ForeachCell::CellArray_global_ghosted& Uin,
                const ForeachCell::CellArray_global_ghosted& Uout,
                real_t dt);

   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
} //namespace muscl_block