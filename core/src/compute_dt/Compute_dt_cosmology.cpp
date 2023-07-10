#include "Compute_dt_base.h"

#include "utils_hydro.h"

#include "states/State_forward.h"

namespace dyablo {


class Compute_dt_cosmology : public Compute_dt
{
public:
  Compute_dt_cosmology( ConfigMap& configMap,
                    ForeachCell& foreach_cell,
                    Timers& timers )
  : cfl( configMap.getValue<real_t>("dt", "cosmo_cfl", 0.5) )
  {}

  void compute_dt( const UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t dt = cfl * scalar_data.get<real_t>("aexp") / scalar_data.get<real_t>("daexp_dt");

    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);

    scalar_data.set<real_t>("dt", dt);
  }

private:
  real_t cfl;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_cosmology, "Compute_dt_cosmology" );
