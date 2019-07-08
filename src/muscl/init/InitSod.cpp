/**
 * \file InitSod.cpp
 * \author Pierre Kestener
 */

#include "InitSod.h"
#include "../SolverHydroMuscl.h"

namespace euler_pablo {
namespace muscl {

// =======================================================
// =======================================================
/**
 * Hydrodynamical shock tube test (SOD).
 *
 * https://en.wikipedia.org/wiki/Sod_shock_tube
 *
 * The following provides python code for the exact solution (EOS is perfect gas)
 * https://github.com/ibackus/sod-shocktube
 *
 * See also https://www3.nd.edu/~gtryggva/CFD-Course2017/Lecture-10-2017.pdf
 * which provides analytical solutions (EOS is perfect gas).
 */
void init_sod(SolverHydroMuscl *psolver)
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
    InitSodRefineFunctor::apply(amr_mesh, configMap, params, level);

    // actually perform refinement
    amr_mesh->adapt();

    // re-compute mesh connectivity (morton index list, nodes coordinates, ...)
    amr_mesh->updateConnectivity();

#if BITPIT_ENABLE_MPI == 1
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
  InitSodDataFunctor::apply(amr_mesh, params, configMap, fm, psolver->U);

} // init_sod

} // namespace muscl

} // namespace euler_pablo
