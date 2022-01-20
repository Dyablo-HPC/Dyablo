#pragma once

#include <memory>

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "muscl_block/io/IOManager_base.h"

class Timers;
class ConfigMap;

namespace dyablo {
namespace muscl_block {

class IOManager_hdf5 : public IOManager{
public: 
  IOManager_hdf5(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers );
  ~IOManager_hdf5();
  void save_snapshot( const ForeachCell::CellArray_global_ghosted& U, uint32_t iter, real_t time );

  struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
} //namespace muscl_block