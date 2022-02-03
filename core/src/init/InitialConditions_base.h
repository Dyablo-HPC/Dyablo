#pragma once 

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo{


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


} // namespace dyablo