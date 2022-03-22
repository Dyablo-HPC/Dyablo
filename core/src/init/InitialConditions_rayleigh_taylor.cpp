#include "InitialConditions_analytical.h"

#include <Kokkos_Random.hpp>

#include "AnalyticalFormula_tools.h"

namespace dyablo{

/**
 * Rayleigh-Taylor instability test.
 * Based on the description of the Athena code :
 * https://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
 **/
struct AnalyticalFormula_RayleighTaylor : public AnalyticalFormula_base{
  using RNGPool = Kokkos::Random_XorShift64_Pool<>;
  using RNGType = RNGPool::generator_type;

  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;

  // Physical parameters
  const real_t rho_top;    // Density at the top of the domain
  const real_t rho_bot;    // Density at the bottom of the domain
  const real_t z0;         // Limit of the transition
  const real_t P0;         // Initial pressure
  const real_t amplitude;  // Amplitude of the perturbation
  const bool   multi_mode; // Number of horizontal modes along the transition
  const int    seed;       // Seed number for the random perturbations
  RNGPool      rand_pool;  // Random pool for multi-mode perturbation

  real_t gz;               // Vertical gravitational acceleration

  AnalyticalFormula_RayleighTaylor( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    
    rho_top(configMap.getValue<real_t>("RT", "rho_top", 2.0)),
    rho_bot(configMap.getValue<real_t>("RT", "rho_bot", 1.0)),
    z0(configMap.getValue<real_t>("RT", "z0", 0.0)),
    P0(configMap.getValue<real_t>("RT", "P0", 2.5)),
    amplitude(configMap.getValue<real_t>("RT", "amplitude", 1.0e-2)),
    multi_mode(configMap.getValue<bool>("RT", "multi_mode", false)),
    seed(configMap.getValue<int>("RT", "seed", 12345)),
    rand_pool(seed*GlobalMpiSession::get_comm_world().MPI_Comm_rank()+1)
  {
    assert(ndim == 2);

    gz = (ndim == 2 ? configMap.getValue<real_t>("gravity", "gy", -0.1)
                    : configMap.getValue<real_t>("gravity", "gz", -0.1));
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
  HydroState3d value(real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz) const 
  {
    HydroState3d res;
    // Overriding vertical direction
    z = (ndim == 2 ? y : z);
    real_t rho, P, u, v, w;
    rho = (z > z0 ? rho_top : rho_bot);
    u = 0;
    P = P0 + rho * gz * z;
    real_t vz;
    if (multi_mode) {
      RNGType rand_gen = rand_pool.get_state();
      vz = amplitude*(rand_gen.drand() - 0.5)*(1.0 + cos(2.0*M_PI*z))*0.5;
      rand_pool.free_state(rand_gen);
    }
    else {
      vz = amplitude*(1.0 + cos(4*M_PI*x))*(1.0 + cos(2.0*M_PI*z))*0.25;
      if (ndim == 3)
        vz *= (1.0 + cos(4.0*M_PI*y));
    }
    if (ndim == 2) {
      v = vz;
      w = 0.0;
    }
    else {
      v = 0.0;
      w = vz;
    }

    real_t Ek = 0.5 * rho * (u*u+v*v+w*w);
    res[ID] = rho;
    res[IU] = rho*u;
    res[IV] = rho*v;
    res[IW] = rho*w;
    res[IE] = Ek + P / (gamma0-1.0);

    return res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_RayleighTaylor>, "rayleigh_taylor");

