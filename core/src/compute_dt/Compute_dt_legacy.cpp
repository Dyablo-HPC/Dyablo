#include "Compute_dt_base.h"
#include "legacy/ComputeDtHydroFunctor.h"

namespace dyablo {


class Compute_dt_legacy : public Compute_dt
{
public:
  Compute_dt_legacy(   ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
  : cfl( configMap.getValue<real_t>("hydro", "cfl", 0.5) ),
    params(configMap),
    pmesh(foreach_cell.get_amr_mesh()),
    foreach_cell(foreach_cell)
  {}

  double compute_dt( const UserData& U_)
  {
    real_t inv_dt = 0;

    LegacyDataArray U(U_);
    static_assert( ForeachCell::has_blocks(), "Legacy solvers are only compatible with block based." );

    ComputeDtHydroFunctor::apply( pmesh.getLightOctree(), 
                                  params, U.get_id2index(),
                                  foreach_cell.blockSize(), U, inv_dt);

    real_t dt = cfl / inv_dt;

    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);

    return dt;
  }

private:
  real_t cfl;
  const ComputeDtHydroFunctor::Params params;
  AMRmesh& pmesh;
  ForeachCell& foreach_cell;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_legacy, "Compute_dt_legacy" );
