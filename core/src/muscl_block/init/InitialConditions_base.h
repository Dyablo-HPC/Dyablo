#pragma once 

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "muscl_block/foreach_cell/ForeachCell.h"

namespace dyablo{
namespace muscl_block{

class InitialConditions{
public:
  // InitialConditions(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,  
  //     Timers& timers);
  virtual void init( ForeachCell::CellArray_global_ghosted& U, const FieldManager& field_manager ) = 0;
  virtual ~InitialConditions(){}
};

using InitialConditionsFactory = RegisteringFactory<InitialConditions, 
  ConfigMap&,
  ForeachCell&, 
  Timers&>;

} // namespace muscl_block
} // namespace dyablo