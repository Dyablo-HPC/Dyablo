/**
 * \file InitIsentropicVortex.cpp
 * \author Maxime Delorme
 **/

#include "InitIsentropicVortex.h"
#include "muscl_block/SolverHydroMusclBlock.h"

namespace dyablo {
namespace muscl_block {

// ================================================
// ================================================
/**
 * Isentropic Vortex test
 *
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 **/

void init_isentropic_vortex(SolverHydroMusclBlock *psolver) {
  std::shared_ptr<AMRmesh> amr_mesh  = psolver->amr_mesh;
  ConfigMap&               configMap = psolver->configMap;
  HydroParams&             params    = psolver->params;

  /* Init global refinement, no parallelism required */
  uint8_t level_min = params.level_min;
  uint8_t level_max = params.level_max;

  for (uint8_t level=0; level<level_min; ++level)
    amr_mesh->adaptGlobalRefine();

#if BITPIT_ENABLE_MPI==1
  // Load balance mesh between mpi processes
  amr_mesh->loadBalance();
#endif

  /* Initial local refinement, mesh only*/
  for (uint8_t level=level_min; level<level_max; ++level) {
    // Mark cells for refinement
    InitIsentropicVortexRefineFunctor::apply(amr_mesh, params, configMap, level);

    // Refine the mesh according to marking
    amr_mesh->adapt();
    
    // Update connectivity
    amr_mesh->updateConnectivity();
  }

  // Now we initialize the data in the mesh

  // Field manager
  auto fm = psolver->fieldMgr.get_id2index();
  
  // Resizing the data
  psolver->resize_solver_data();

  // And init
  InitIsentropicVortexDataFunctor::apply(amr_mesh, params, configMap, fm,
					 psolver->blockSizes, psolver->Uhost);

  // Finally copy data to device
  Kokkos::deep_copy(psolver->U, psolver->Uhost);
} // init_isentropic_vortex
  
} // namespace muscl_block
} // namespace dyablo
