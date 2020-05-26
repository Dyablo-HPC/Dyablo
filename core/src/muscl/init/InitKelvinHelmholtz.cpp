/**
 * \file InitKelvinHelmholtz.cpp
 * \author Pierre Kestener
 */

#include "InitKelvinHelmholtz.h"
#include "../SolverHydroMuscl.h"

namespace dyablo {
namespace muscl {

// =======================================================
// =======================================================
/**
 * Hydrodynamical Kelvin-Helmholtz instability Test.
 *
 * see https://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
 *
 * See also article by Robertson et al:
 * "Computational Eulerian hydrodynamics and Galilean invariance", 
 * B.E. Robertson et al, Mon. Not. R. Astron. Soc., 401, 2463-2476, (2010).
 *
 */
void init_kelvin_helmholtz(SolverHydroMuscl *psolver)
{

  std::shared_ptr<AMRmesh> amr_mesh = psolver->amr_mesh;
  ConfigMap &configMap = psolver->configMap;
  HydroParams& params = psolver->params;

  /*
   * this is the initial global refine, to reach level_min / no parallelism
   * so far (every MPI process does that)
   */
  int level_min = params.level_min;
  int level_max = params.level_max;  

  for (int iter=0; iter<level_min; iter++) {
    amr_mesh->adaptGlobalRefine();
  }
#if BITPIT_ENABLE_MPI==1
  // (Load)Balance the octree over the MPI processes.
  amr_mesh->loadBalance();
#endif

  // after the global refine stages, all cells are at level = level_min

  // genuine initial refinement
  for (int level=level_min; level<level_max; ++level) {

    // mark cells for refinement
    InitKelvinHelmholtzRefineFunctor::apply(amr_mesh, configMap, params, level);

    // actually perform refinement
    amr_mesh->adapt();

    // re-compute mesh connectivity (morton index list, nodes coordinates, ...)
    amr_mesh->updateConnectivity();

#if BITPIT_ENABLE_MPI==1
    // (Load)Balance the octree over the MPI processes.
    amr_mesh->loadBalance();
#endif

  } // end for level

  // field manager index array
  auto fm = psolver->fieldMgr.get_id2index();

  psolver->resize_solver_data();

  /*
   * perform user data init
   */
  InitKelvinHelmholtzDataFunctor::apply(amr_mesh, params, configMap, fm, psolver->U);


} // init_kelvin_helmholtz

} // namespace muscl

} // namespace dyablo
