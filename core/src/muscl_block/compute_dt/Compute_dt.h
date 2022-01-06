#pragma once

#include "muscl_block/ComputeDtHydroFunctor.h"
#include "muscl_block/foreach_cell/AMRBlockForeachCell_group.h"

namespace dyablo {
namespace muscl_block {

class Compute_dt
{
  using CellArray = AMRBlockForeachCell_group::CellArray_global_ghosted;

public:
  Compute_dt(   ConfigMap& configMap,
                const HydroParams& params, 
                AMRmesh& pmesh,
                Timers& timers )
  : cfl( configMap.getFloat("hydro", "cfl", 0.5) ),
    configMap(configMap),
    params(params),
    pmesh(pmesh)
  {}

  double compute_dt( const CellArray& U)
  {
    real_t inv_dt = 0;

    ComputeDtHydroFunctor::apply( pmesh.getLightOctree(), 
                                  configMap,
                                  params, U.fm,
                                  {U.bx,U.by,U.bz}, U.U, inv_dt);

    real_t dt = cfl / inv_dt;

    assert(dt>0);

    return dt;
  }

private:
  real_t cfl;
  const ConfigMap& configMap;
  const HydroParams params;
  AMRmesh& pmesh;
};

} // namespace muscl_block
} // namespace dyablo 