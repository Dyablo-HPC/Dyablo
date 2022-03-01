#include "InitialConditions_analytical.h"

#include "AnalyticalFormula_tools.h"

namespace dyablo{

struct AnalyticalFormula_implode : public AnalyticalFormula_base{
  
  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;

  const real_t x0; // Position at which the internal limit intercepts y=0
  // Variables inside the diamond
  const real_t p_in;
  const real_t p_out;
  // Variables outside the diamond
  const real_t rho_in;
  const real_t rho_out;

  AnalyticalFormula_implode( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    
    x0(configMap.getValue<real_t>("implode", "x0", 0.25)),
    p_in(configMap.getValue<real_t>("implode", "p_in", 0.14)),
    p_out(configMap.getValue<real_t>("implode", "p_out", 1.0)),
    rho_in(configMap.getValue<real_t>("implode", "rho_in", 0.125)),
    rho_out(configMap.getValue<real_t>("implode", "rho_out", 1.0))
  {
    assert(ndim == 2);
  }

  KOKKOS_INLINE_FUNCTION
  bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const 
  {
    real_t gamma0 = this->gamma0;
    real_t smallr = this->smallr;
    real_t smallp = this->smallp;
    real_t error_max = this->error_max;
    return AnalyticalFormula_tools::auto_refine( *this, gamma0, smallr, smallp, error_max,
                                                  x, y, z, dx, dy, dz );
  }

  KOKKOS_INLINE_FUNCTION
  HydroState3d value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    bool inside = (y < x-x0);
    real_t p, rho;
    if (inside) {
      p = p_in;
      rho = rho_in;
    }
    else{
      p = p_out;
      rho = rho_out;
    }
    HydroState3d res;
    res[ID] = rho;
    res[IE] = p / (gamma0-1.0);

    return res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_implode>, "implode");

