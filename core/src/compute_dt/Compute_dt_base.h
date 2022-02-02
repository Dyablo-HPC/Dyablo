#pragma once

#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {
namespace muscl_block {

class Compute_dt{
public: 
  // Compute_dt(
  //               const ConfigMap& configMap,
  //               ForeachCell& foreach_cell,
  //               Timers& timers );
  virtual ~Compute_dt(){}
  virtual double compute_dt( const ForeachCell::CellArray_global_ghosted& Uin) = 0;
};

using Compute_dtFactory = RegisteringFactory< Compute_dt, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
} //namespace muscl_block