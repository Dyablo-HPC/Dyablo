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
                const ConfigMap& configMap,
                const HydroParams& params, 
                AMRmesh& pmesh,
                const id2index_t& fm,
                uint32_t bx, uint32_t by, uint32_t bz,
                Timers& timers );
  ~IOManager_hdf5();
  void save_snapshot( const DataArrayBlock& U, const DataArrayBlock& Ughost, uint32_t iter, real_t time );

  struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
} //namespace muscl_block