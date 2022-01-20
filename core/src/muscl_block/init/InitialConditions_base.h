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
  //     AMRmesh& pmesh,
  //     FieldManager fieldMgr,
  //     uint32_t nbOctsPerGroup,  
  //     uint32_t bx, uint32_t by, uint32_t bz,  
  //     Timers& timers);
  virtual void init( ForeachCell::CellArray_global_ghosted& U ) = 0;
  virtual ~InitialConditions(){}
};

using InitialConditionsFactory = RegisteringFactory<InitialConditions, 
  ConfigMap&,
  AMRmesh&,
  FieldManager,
  uint32_t,  
  uint32_t, uint32_t, uint32_t,  
  Timers&>;

} // namespace muscl_block
} // namespace dyablo