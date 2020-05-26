/**
 * \file InitKelvinHelmholtz.cpp
 * \author Maxime Delorme
 */

#include "InitKelvinHelmholtz.h"
#include "../SolverHydroMusclBlock.h"

namespace dyablo {
namespace muscl_block {

// ================================================
// ================================================
/**
 * Hydrodynamical Kelvin-Helmholtz instability Test
 *
 * see https://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
 *
 * See also article by Robertson et al:
 * "Computational Eulerian hydrodynamics and Galilean invariance", 
 * B.E. Robertson et al, Mon. Not. R. Astron. Soc., 401, 2463-2476, (2010).
 *
 */
void init_kelvin_helmholtz(SolverHydroMusclBlock *psolver) {
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
    InitKelvinHelmholtzRefineFunctor::apply(amr_mesh, configMap, params, level);

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
  
  InitKelvinHelmholtzDataFunctor::apply(amr_mesh, params, configMap, fm,
					psolver->blockSizes, psolver->Uhost);
  // Upload data on device
  Kokkos::deep_copy(psolver->U, psolver->Uhost);
} // init_kelvin_helmholtz
} // namespace muscl_block
} // namespace dyablo
