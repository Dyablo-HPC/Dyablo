/**
 * \file InitGreshoVortex.cpp
 * \author Pierre Kestener
 */

//#include "InitGreshoVortex.h"
#include "../SolverHydroMuscl.h"

namespace euler_pablo {
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

  // TODO
  std::cerr << "Implement me !\n";

} // init_gresho_vortex

} // namespace muscl

} // namespace euler_pablo
