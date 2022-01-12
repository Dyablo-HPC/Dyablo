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

class MusclBlockUpdate_generic : public MusclBlockUpdate{
public: 
  MusclBlockUpdate_generic(
                ConfigMap& configMap,
                const HydroParams& params, 
                AMRmesh& pmesh,
                const id2index_t& fm,
                uint32_t bx, uint32_t by, uint32_t bz,
                Timers& timers );
  ~MusclBlockUpdate_generic();
  void update(  DataArrayBlock U, DataArrayBlock Ughost,
                DataArrayBlock Uout, 
                real_t dt);

   struct Data;
private:
  std::unique_ptr<Data> pdata;
};

} //namespace dyablo 
} //namespace muscl_block