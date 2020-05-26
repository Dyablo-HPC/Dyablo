/**
 * \file InitImplode.cpp
 * \author Pierre Kestener
 */

#include "InitImplode.h"
#include "../SolverHydroMusclBlock.h"

namespace dyablo {
namespace muscl_block {

// =======================================================
// =======================================================
/**
 * Hydrodynamical implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 *
 * Initial condition is mostly done on host, the final refined initial
 * condition data are uploaded to kokkos device.
 */
void init_implode(SolverHydroMusclBlock *psolver)
{

  std::shared_ptr<AMRmesh> amr_mesh = psolver->amr_mesh;
  ConfigMap&   configMap = psolver->configMap;
  HydroParams& params    = psolver->params;
  
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
  for (int level=level_min; level<level_max; ++level) {

    // mark cells for refinement
    InitImplodeRefineFunctor::apply(amr_mesh, configMap, params, level);

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

  // now we know the size of the mesh, we can allocate memory for
  // heavy data (U, U2, Uhost, ...)
  psolver->resize_solver_data();

  /*
   * perform user data init
   */
  InitImplodeDataFunctor::apply(amr_mesh, params, configMap, fm, 
                                psolver->blockSizes,
                                psolver->Uhost);

  // upload data on device
  Kokkos::deep_copy(psolver->U, psolver->Uhost);

} // init_implode

} // namespace muscl_block

} // namespace dyablo
