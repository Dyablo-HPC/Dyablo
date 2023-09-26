#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "enums.h"

namespace dyablo {

class ParabolicUpdate {
public: 
  virtual ~ParabolicUpdate(){}

  virtual void update(UserData &U,
                      ScalarSimulationData& scalar_data) = 0;
};

using ParabolicUpdateFactory = RegisteringFactory< ParabolicUpdate, 
  ConfigMap&,       /*configMap*/
  ForeachCell&,     /*foreach_cell*/
  Timers&,          /*timers*/
  ParabolicTermType /*term_type*/>;  

} //namespace dyablo 
