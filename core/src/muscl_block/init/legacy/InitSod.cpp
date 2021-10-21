/**
 * \file InitSod.cpp
 * \author Maxime Delorme
 **/

#include "InitSod.h"
#include "muscl_block/SolverHydroMusclBlock.h"

namespace dyablo {
namespace muscl_block {

// ================================================
// ================================================
/**
 * Sod Shock tube in 2 or 3 dimensions
 **/
void init_sod(SolverHydroMusclBlock *psolver) {
  std::shared_ptr<AMRmesh> amr_mesh  = psolver->amr_mesh;
  ConfigMap&               configMap = psolver->configMap;
  HydroParams&             params    = psolver->params;
  
  /**
   * Initial global refinement
   **/
  uint8_t level_min = params.level_min;
  uint8_t level_max = params.level_max;
  
  for (uint8_t iter=0; iter < level_min; ++iter)
    amr_mesh->adaptGlobalRefine();

#if BITPIT_ENABLE_MPI==1
  amr_mesh->loadBalance();
#endif

  /** 
   * Mesh refinement depending on the ICs
   **/
  for (uint8_t level=level_min; level < level_max; ++level) {
    // Mark cells for refinement
    InitSodRefineFunctor::apply(amr_mesh, params, configMap, level);

    // Refine current level and update connectivity
    amr_mesh->adapt();
    amr_mesh->updateConnectivity();

#if BITPIT_ENABLE_MPI==1
    amr_mesh->loadBalance();
#endif
  } // end for level

  /**
   * Now we initialize the blocks to the ICs
   **/
  auto fm = psolver->fieldMgr.get_id2index();

  // We resize the solver arrays
  psolver->resize_solver_data();

  // And we initialize the blocks
  InitSodDataFunctor::apply(amr_mesh, params, configMap, fm, psolver->blockSizes, psolver->Uhost);

  // Finally we copy the data on device
  Kokkos::deep_copy(psolver->U, psolver->Uhost);
}
  
} // namespace muscl_block
} // namespace dyablo
