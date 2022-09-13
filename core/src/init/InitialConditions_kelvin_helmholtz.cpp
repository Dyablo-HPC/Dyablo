#include "InitialConditions_analytical.h"

#include "AnalyticalFormula_tools.h"

namespace dyablo{

/**
 * Kelvin-Helmholtz instability test.
 * Based on Lecoanet et al "A validated non-linear kelvin-helmholtz benchmark for numerical 
 * hydrodynamics", 2016, Monthly Notices of the Royal Astronomy Society
 **/
struct AnalyticalFormula_KelvinHelmholtz : public AnalyticalFormula_base{
  
  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;

  // Physical parameters
  const real_t rho_fac; // delta rho / rho0 parameter
  const real_t z1;      // Lower limit for the transition
  const real_t z2;      // Upper limit for the transition
  const real_t a;       // Smoothing factor for the transition
  const real_t A;       // Amplitude of the velocity variation
  const real_t sigma;   // Smoothing factor for the vertical velocity transition
  const real_t P0;      // Base pressure
  const real_t uflow;   // Base horizontal velocity

  AnalyticalFormula_KelvinHelmholtz( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    
    rho_fac(configMap.getValue<real_t>("KH", "rho_fac", 0.0)),
    z1(configMap.getValue<real_t>("KH", "z1", 0.5)),
    z2(configMap.getValue<real_t>("KH", "z2", 1.5)),
    a(configMap.getValue<real_t>("KH", "a", 0.05)),
    A(configMap.getValue<real_t>("KH", "A", 0.01)),
    sigma(configMap.getValue<real_t>("KH", "sigma", 0.2)),
    P0(configMap.getValue<real_t>("KH", "P0", 1.0)),
    uflow(configMap.getValue<real_t>("KH", "uflow", 1.0))
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
  ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    ConsHydroState res;
    const real_t q1 = tanh((y-z1)/a);
    const real_t q2 = tanh((y-z2)/a);
    const real_t s2 = sigma*sigma;
    const real_t dz1 = (y-z1)*(y-z1);
    const real_t dz2 = (y-z2)*(y-z2);
    const real_t rho = 1.0 + rho_fac*0.5*(q1 - q2);
    const real_t u   = uflow*(q1 - q2 - 1.0);
    const real_t v   = A*sin(2.0*M_PI*x)*(exp(-dz1/s2) + exp(-dz2/s2));
    const real_t Ek  = rho*0.5*(u*u+v*v);

    res.rho   = rho;
    res.rho_u = rho*u;
    res.rho_v = rho*v;
    res.e_tot = Ek + P0 / (gamma0-1.0);

    return res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_KelvinHelmholtz>, 
                 "kelvin_helmholtz");

