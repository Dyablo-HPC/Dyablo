#pragma once

#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "UserData.h"

namespace dyablo {


class Compute_dt{
public: 
  // Compute_dt(
  //               const ConfigMap& configMap,
  //               ForeachCell& foreach_cell,
  //               Timers& timers );
  virtual ~Compute_dt(){}
  virtual double compute_dt( const UserData& Uin) = 0;
};

using Compute_dtFactory = RegisteringFactory< Compute_dt, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
