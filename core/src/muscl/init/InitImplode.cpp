/**
 * \file InitImplode.cpp
 * \author Pierre Kestener
 */

#include "InitImplode.h"
#include "../SolverHydroMuscl.h"

namespace dyablo {
namespace muscl {

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
void init_implode(SolverHydroMuscl *psolver)
{

  std::shared_ptr<AMRmesh> amr_mesh = psolver->amr_mesh;
  ConfigMap &configMap = psolver->configMap;
  HydroParams& params = psolver->params;

  // field manager index array
  auto fm = psolver->fieldMgr.get_id2index();
  
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

  // re-compute mesh connectivity (morton index list, nodes coordinates, ...)
  amr_mesh->updateConnectivity();

  // after the global refine stages, all cells are at level = level_min

  // initialize user data (U and U2) at level_min
  Kokkos::resize(psolver->U,amr_mesh->getNumOctants(),params.nbvar);
  Kokkos::resize(psolver->U2,amr_mesh->getNumOctants(),params.nbvar);
  InitImplodeDataFunctor::apply(amr_mesh, params, configMap, fm, psolver->U);
  Kokkos::deep_copy(psolver->U2, psolver->U);
  
  // update mesh until we reach level_max
  for (int level = level_min; level < level_max; ++level) {
    
    psolver->do_amr_cycle();
    
    // re-compute U on the new mesh
    InitImplodeDataFunctor::apply(amr_mesh, params, configMap, fm, psolver->U);
    Kokkos::deep_copy(psolver->U2,psolver->U);
    
  }

} // init_implode

} // namespace muscl

} // namespace dyablo