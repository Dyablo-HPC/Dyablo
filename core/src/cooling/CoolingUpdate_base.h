#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "ScalarSimulationData.h"
#include "UserData.h"

namespace dyablo {


class CoolingUpdate{
public: 
  virtual ~CoolingUpdate(){}
  
  virtual void update( UserData &U,
                       ScalarSimulationData& scalar_data) = 0;
};

using CoolingUpdateFactory = RegisteringFactory< CoolingUpdate, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
