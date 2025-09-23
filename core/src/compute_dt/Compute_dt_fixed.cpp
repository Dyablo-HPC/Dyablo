#include "Compute_dt_base.h"

#include "utils_hydro.h"

#include "states/State_forward.h"

namespace dyablo {

/**
 * @brief Returns a fixed value for the time-stepping
 **/
class Compute_dt_fixed : public Compute_dt
{
public:
  Compute_dt_fixed( ConfigMap& configMap,
                    ForeachCell& foreach_cell,
                    Timers& timers )
  : foreach_cell(foreach_cell),
    fixed_dt( configMap.getValue<real_t>("dt", "fixed_dt", 1.0e-2))
  { }

  void compute_dt( UserData& U, ScalarSimulationData& scalar_data )
  {
    scalar_data.set<real_t>("dt", fixed_dt);
  }

private:
  ForeachCell& foreach_cell;

  real_t fixed_dt;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_fixed, "Compute_dt_fixed" );
