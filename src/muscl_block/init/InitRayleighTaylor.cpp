/**
 * \file InitRayleighTaylor.cpp
 * \author Maxime Delorme
 */

#include "InitRayleighTaylor.h"
#include "../SolverHydroMusclBlock.h"

namespace dyablo {
namespace muscl_block {

using DataFunctor   = InitRayleighTaylorDataFunctor;
using RefineFunctor = InitRayleighTaylorRefineFunctor;

// ================================================
// ================================================
/**
 * Rayleigh-Taylor instability as per https://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
 * References:
 *  - Liska, R., Wendroff, B. "Comparison of Several Difference Schemes on 1D and 2D Test Problems for the Euler Equations", SIAM, J. Sci. Comput., 25(3), 995-1017, 2003
 *  - Jun, B.I., Norman, M.L. "A Numerical Study of Rayleigh-Taylor instability in Magnetic Fluids", ApJ, 453:332-349, 1995
 */
void init_rayleigh_taylor(SolverHydroMusclBlock *psolver) {
  std::shared_ptr<AMRmesh> amr_mesh  = psolver->amr_mesh;
  ConfigMap&               configMap = psolver->configMap;
  HydroParams&             params    = psolver->params;

  /* Initial global refinement, no parallelism required */
  int level_min = params.level_min;
  int level_max = params.level_max;

  if (params.gravity_type != GRAVITY_CONSTANT) {
    std::cerr << "ERROR: Gravity type should be set to constant for Rayleigh-Taylor instability" << std::endl;
    // HERE do something like std::exit(1);
  }

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
} // init_RayleighTaylor
} // namespace muscl_block
} // namespace dyablo
