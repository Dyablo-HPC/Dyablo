#pragma once

#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "update/MusclBlockUpdate_base.h"

class Timers;
class ConfigMap;

namespace dyablo {
namespace muscl_block {

class MusclBlockUpdate_generic : public MusclBlockUpdate{
public: 
  MusclBlockUpdate_generic(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers );
  ~MusclBlockUpdate_generic();
  void update(  const ForeachCell::CellArray_global_ghosted& Uin,
                const ForeachCell::CellArray_global_ghosted& Uout,
                real_t dt);
   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
} //namespace muscl_block