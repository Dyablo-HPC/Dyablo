/**
 * \file InitTemplate.cpp
 * \author Maxime Delorme
 */

/**
 * This file is a template for the construction of a new problem.
 * It should be copied to the correct name (e.g. InitBlast.cpp) and adapted
 * so that the right functors are called.
 *
 * Please follow the instructions in InitTemplate.h.
 **/

#include "InitXXXXX.h"
#include "../SolverHydroMusclBlock.h"

namespace dyablo {
namespace muscl_block {

// Redefine these to the functors you have setup in InitXXXXX.h
using DataFunctor   = InitXXXXXDataFunctor;
using RefineFunctor = InitXXXXXRefineFunctor;

// ================================================
// ================================================
/**
 * Here add a short description of your test and potential references
 * to bibliographic material
 */
void init_XXXXX(SolverHydroMusclBlock *psolver) {
  std::shared_ptr<AMRmesh> amr_mesh  = psolver->amr_mesh;
  ConfigMap&               configMap = psolver->configMap;
  HydroParams&             params    = psolver->params;

  /* Initial global refinement, no parallelism required */
  int level_min = params.level_min;
  int level_max = params.level_max;

  for (uint8_t level=0; level<level_min; ++level)
    amr_mesh->adaptGlobalRefine();

#if BITPIT_ENABLE_MPI==1
  // Load balance the octree over MPI processes
  amr_mesh->loadBalance();
#endif

  /* Initial local refinement, mesh only */
  for (uint8_t level=level_min; level<level_max; ++level) {
    // Mark cells for refinement
    RefineFunctor::apply(amr_mesh, configMap, params, level);

    // Refine the mesh according to marking
    amr_mesh->adapt();

    // Update connectivity
    amr_mesh->updateConnectivity();

#if BITPIT_ENABLE_MPI==1
    amr_mesh->loadBalance();
#endif
  }

  /* Now that everything is refined, we initialize the data */

  // Field manager
  auto fm = psolver->fieldMgr.get_id2index();

  // Resizing the data to the size of the mesh
  psolver->resize_solver_data();

  // And data init
  DataFunctor::apply(amr_mesh, params, configMap, fm, psolver->blockSizes, psolver->Uhost);
  // Upload data on device
  Kokkos::deep_copy(psolver->U, psolver->Uhost);
} // init_XXXXX
} // namespace muscl_block
} // namespace dyablo
