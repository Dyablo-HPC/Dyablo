#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "UserData.h"

namespace dyablo {

/**
 * Base class for MapUserData plugin
 * Contains the remap() method to fill new mesh after AMR refinement/corasening 
 **/
class MapUserData{
public: 
  // MapUserData(
  //     ConfigMap& configMap,
  //     ForeachCell& foreach_cell,
  //     Timers& timers );
  virtual ~MapUserData(){}

  /**
   * Save a snapshot of the current AMR mesh in ForeachCell
   * This should be called before calling AMRmesh::refine()
   **/
  virtual void save_old_mesh() = 0;
  
  /**
   * Fill new array Uout from Uin
   * @param Uin A cell array on the old amr mesh (mesh at last save_old_mesh() call) to copy data from
   * @param Uout An array to store new cells for the new mesh to write new mesh data to
   **/
  virtual void remap( UserData& Uin ) = 0;
};

using MapUserDataFactory = RegisteringFactory< MapUserData, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 