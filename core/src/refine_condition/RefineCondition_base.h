#pragma once

#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {
namespace muscl_block {

class RefineCondition{
public: 
  // RefineCondition(
  //               const ConfigMap& configMap,
  //               ForeachCell& foreach_cell,
  //               Timers& timers );
  virtual ~RefineCondition(){}
  virtual void mark_cells( const ForeachCell::CellArray_global_ghosted& U ) = 0;
};

using RefineConditionFactory = RegisteringFactory< RefineCondition, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
} //namespace muscl_block