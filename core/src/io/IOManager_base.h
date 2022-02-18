#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {


/**
 * Base class for IO manager
 * 
 * IO manager is configured and instanciated during initialisation with the IOManagerFactory 
 * save_snapshot() can then be called at any timestep to output a snapshot
 **/
class IOManager{
public: 
  // IOManager(
  //               ConfigMap& configMap,
  //               ForeachCell& foreach_cell,
  //               Timers& timers );
  virtual ~IOManager(){}

  /**
   * Output simulation snapshot to a file
   * @param U, Ughost : Simulation data
   * @param iter iteration number
   * @param time physical time of the simulation
   **/
  virtual void save_snapshot( const ForeachCell::CellArray_global_ghosted& U_, uint32_t iter, real_t time ) = 0;
};

using IOManagerFactory = RegisteringFactory< IOManager, 
  ConfigMap& /*configMap*/,
  ForeachCell& /*foreach_cell*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
