/**
 * \file InitRayleighTaylor.cpp
 * \author Pierre Kestener
 */

//#include "InitRayleighTaylor.h"
#include "../SolverHydroMuscl.h"

namespace dyablo {
namespace muscl {

// =======================================================
// =======================================================
void init_rayleigh_taylor(SolverHydroMuscl *psolver)
{

  //std::shared_ptr<AMRmesh> amr_mesh = psolver->amr_mesh;
  //ConfigMap &configMap = psolver->configMap;
  //HydroParams& params = psolver->params;

  // TODO when gravity is implemented in the cell-based
  assert(false); // Not implemented yet

} // init_rayleigh_taylor

} // namespace muscl

} // namespace dyablo
