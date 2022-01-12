#pragma once

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "utils/misc/RegisteringFactory.h"
#include "utils/monitoring/Timers.h"

namespace dyablo {
namespace muscl_block {

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
  //               AMRmesh& pmesh,
  //               const id2index_t& fm,
  //               uint32_t bx, uint32_t by, uint32_t bz,
  //               Timers& timers );
  virtual ~IOManager(){}

  /**
   * Output simulation snapshot to a file
   * @param U, Ughost : Simulation data
   * @param iter iteration number
   * @param time physical time of the simulation
   **/
  virtual void save_snapshot( const DataArrayBlock& U, const DataArrayBlock& Ughost, uint32_t iter, real_t time ) = 0;
};

using IOManagerFactory = RegisteringFactory< IOManager, 
  ConfigMap& /*configMap*/,
  AMRmesh& /*pmesh*/,
  const FieldManager& /*fm*/,
  uint32_t /*bx*/, uint32_t /*by*/, uint32_t /*bz*/,
  Timers& /*timers*/ >;

} //namespace dyablo 
} //namespace muscl_block