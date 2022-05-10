#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {


class MapUserData{
public: 
  // MapUserData(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,
  //     Timers& timers );
  virtual ~MapUserData(){}
  virtual void save_old_mesh() = 0;
  virtual void remap(   const ForeachCell::CellArray_global_ghosted& Uin,
                        const ForeachCell::CellArray_global_ghosted& Uout ) = 0;
};

using MapUserDataFactory = RegisteringFactory< MapUserData, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 