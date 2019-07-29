/**
 * \file InitGreshoVortex.cpp
 * \author Pierre Kestener
 */

#include "InitGreshoVortex.h"
#include "../SolverHydroMuscl.h"

namespace dyablo {
namespace muscl {

// =======================================================
// =======================================================
/**
 * Hydrodynamical Gresho vortex Test.
 *
 * \sa https://www.cfd-online.com/Wiki/Gresho_vortex
 * \sa https://arxiv.org/abs/1409.7395 - section 4.2.3
 * \sa https://arxiv.org/abs/1612.03910
 *
 */
void init_gresho_vortex(SolverHydroMuscl *psolver)
{

  std::shared_ptr<AMRmesh> amr_mesh = psolver->amr_mesh;
  ConfigMap &configMap = psolver->configMap;
  HydroParams& params = psolver->params;

  bool use_geometric_refinement = configMap.getBool("Gresho", "use_geometric_refinement", false);

  // field manager index array
  auto fm = psolver->fieldMgr.get_id2index();

  /*
   * this is the initial global refine, 
   * to reach level_min / no parallelism
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

  // now decide which initial refinement we want:
  // - based of geometric information
  // - based on data
  if (use_geometric_refinement) {

    for (int level=level_min; level<level_max; ++level) {

      // mark cells for refinement
      InitGreshoVortexRefineFunctor::apply(amr_mesh, configMap, params, level);
      
      // actually perform refinement
      amr_mesh->adapt();

      // re-compute mesh connectivity (morton index list, nodes coordinates, ...)
      amr_mesh->updateConnectivity();

#if BITPIT_ENABLE_MPI==1
      // (Load)Balance the octree over the MPI processes.
      amr_mesh->loadBalance();
#endif
      
    } // end for level

    psolver->resize_solver_data();
    
    /*
     * perform user data initialization
     */
    InitGreshoVortexDataFunctor::apply(amr_mesh, params, configMap, fm, psolver->U);

  } else { // refine use regular refinement criterium

    // re-compute mesh connectivity (morton index list, nodes coordinates, ...)
    amr_mesh->updateConnectivity();

    // after the global refine stages, all cells are at level = level_min

    // initialize user data (U and U2) at level_min
    Kokkos::resize(psolver->U,amr_mesh->getNumOctants(),params.nbvar);
    Kokkos::resize(psolver->U2,amr_mesh->getNumOctants(),params.nbvar);
    InitGreshoVortexDataFunctor::apply(amr_mesh, params, configMap, fm, psolver->U);
    Kokkos::deep_copy(psolver->U2, psolver->U);

    // update mesh until we reach level_max
    for (int level = level_min; level < level_max; ++level) {

      psolver->do_amr_cycle();

      // re-compute U on the new mesh
      InitGreshoVortexDataFunctor::apply(amr_mesh, params, configMap, fm,
                                         psolver->U);
      Kokkos::deep_copy(psolver->U2, psolver->U);
    }
  }

} // init_gresho_vortex

} // namespace muscl

} // namespace dyablo
