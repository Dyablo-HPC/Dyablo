#include "Compute_dt_base.h"

#include "Cosmo.h"

namespace dyablo {


class Compute_dt_cosmology : public Compute_dt
{
public:
  Compute_dt_cosmology( ConfigMap& configMap,
                    ForeachCell& foreach_cell,
                    Timers& timers )
  : cosmo_manager( configMap )
  {}

  void compute_dt( const UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t aexp = scalar_data.get<real_t>("aexp");
    real_t dt = cosmo_manager.compute_cosmo_dt( aexp );

    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);

    scalar_data.set<real_t>("dt", dt);
  }

private:
  CosmoManager cosmo_manager;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_cosmology, "Compute_dt_cosmology" );
