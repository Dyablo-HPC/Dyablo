#pragma once

#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "UserData.h"
#include "ScalarSimulationData.h"

namespace dyablo {


class RefineCondition{
public: 
  enum Marker
  {
    COARSEN = -1,
    NOCHANGE = 0,
    REFINE = 1
  };
  // RefineCondition(
  //               const ConfigMap& configMap,
  //               ForeachCell& foreach_cell,
  //               Timers& timers );
  virtual ~RefineCondition(){}
  virtual void mark_cells( const UserData& U, ScalarSimulationData& scalar_data) = 0;
};

using RefineConditionFactory = RegisteringFactory< RefineCondition, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
