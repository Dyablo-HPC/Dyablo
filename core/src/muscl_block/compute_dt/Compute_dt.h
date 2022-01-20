#pragma once

#include "muscl_block/ComputeDtHydroFunctor.h"

namespace dyablo {
namespace muscl_block {

class Compute_dt
{
public:
  Compute_dt(   ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
  : cfl( configMap.getValue<real_t>("hydro", "cfl", 0.5) ),
    params(configMap),
    pmesh(foreach_cell.pmesh)
  {}

  double compute_dt( const ForeachCell::CellArray_global_ghosted& U)
  {
    real_t inv_dt = 0;

    ComputeDtHydroFunctor::apply( pmesh.getLightOctree(), 
                                  params, U.fm,
                                  {U.bx,U.by,U.bz}, U.U, inv_dt);

    real_t dt = cfl / inv_dt;

    assert(dt>0);

    return dt;
  }

private:
  real_t cfl;
  const ComputeDtHydroFunctor::Params params;
  AMRmesh& pmesh;
};

} // namespace muscl_block
} // namespace dyablo 