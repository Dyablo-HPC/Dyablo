/**
 * \file InitFourQuadrant.cpp
 * \author Maxime Delorme 
 * \author Pierre Kestener
 */

#include "InitFourQuadrant.h"
#include "muscl_block/SolverHydroMusclBlock.h"

namespace dyablo {
namespace muscl_block {

// =======================================================
// =======================================================
/**
 * Init four quadrant (piecewise constant).
 *
 * Four quadrant 2D riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
void init_four_quadrant(SolverHydroMusclBlock *psolver)
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
    //std::cout << "MPI rank=" << amr_mesh->getRank() << " | NB cells =" << amr_mesh->getNumOctants() << "\n";

  // after the global refine stages, all cells are at level = level_min

  // genuine initial refinement
  for (int level = level_min; level < level_max; ++level) {

    // mark cells for refinement
    InitFourQuadrantRefineFunctor::apply(amr_mesh, configMap, params, level);

    // actually perform refinement
    amr_mesh->adapt();

    // re-compute mesh connectivity (morton index list, nodes coordinates, ...)
    amr_mesh->updateConnectivity();

#if BITPIT_ENABLE_MPI == 1
    // (Load)Balance the octree over the MPI processes.
    amr_mesh->loadBalance();
#endif

  } // end for level

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = psolver->fieldMgr.get_id2index();

  psolver->resize_solver_data();

  /*
   * perform user data init
   */
  InitFourQuadrantDataFunctor::apply(amr_mesh, params, configMap, fm, psolver->blockSizes, psolver->Uhost);
  
  // upload data on device
  Kokkos::deep_copy(psolver->U, psolver->Uhost);
} // init_four_quadrant

} // namespace muscl_block

} // namespace dyablo
