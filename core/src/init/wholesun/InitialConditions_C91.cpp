#include "../InitialConditions_analytical.h"

#include "../AnalyticalFormula_tools.h"
#include "states/State_forward.h"

#include <Kokkos_Random.hpp>

namespace dyablo{

/**
 * @brief Setup based on the paper by Cattaneo et al 1991
 * "Turbulent compressible compression"
 * F. Cattaneo, N. Brummell, J. Toomre; The Astrophysical Journal, 1991, vol 370(9)
 * 
 * This setup initializes the domain as a single polytrope in hydrostatic equilibrium with 
 * a small pressure perturbation. By cooling the top of the domain, and heating the bottom
 * the setup generates solar-like convection plumes.
 */
struct AnalyticalFormula_C91 : public AnalyticalFormula_base{
  using RNGPool = Kokkos::Random_XorShift64_Pool<>;
  using RNGType = RNGPool::generator_type;

  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;

  const int    seed;           // Seed number for the random perturbations
  const real_t theta;          // Temperature gradient
  const real_t pert_amplitude; // Amplitude of the perturbations in pressure
  const real_t m;              // Polytropic index
  const real_t rho0, T0;       // Hydro values at the top layer
  RNGPool      rand_pool;      // Random pool for multi-mode perturbation
  bool geom_ref;

  AnalyticalFormula_C91( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    seed(configMap.getValue<int>("C91", "seed", 12345)),
    theta(configMap.getValue<real_t>("C91", "theta", 10.0)),
    pert_amplitude(configMap.getValue<real_t>("C91", "pert_amplitude", 1.0e-3)),
    m(configMap.getValue<real_t>("C91", "mpoly", 1.0)),
    rho0(configMap.getValue<real_t>("C91", "rho0", 10.0)),
    T0(configMap.getValue<real_t>("C91", "T0", 10.0)),
    rand_pool(seed*GlobalMpiSession::get_comm_world().MPI_Comm_rank()+1)
  {
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
    // Hydro init
    real_t d = (ndim == 2 ? y : z);
    real_t T   = T0 + theta * d;
    real_t rho = rho0    * pow(T/T0, m);
    real_t p   = rho0*T0 * pow(T/T0, m+1.0);
    
    // Perturbation
    RNGType rand_gen = rand_pool.get_state();
    auto pert = pert_amplitude * ((rand_gen.drand() * 2.0) - 1.0);
    rand_pool.free_state(rand_gen);
    
    // No perturbation in the limit layer to avoid interaction with the BCs
    if (d < 0.1 || d > 0.9)
      pert = 0.0;  

    ConsHydroState res{};

    // Assigning the conservative variable as a result
    res.rho = rho;
    res.e_tot = p * (1.0+pert) / (gamma0-1.0);

    return res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_C91>, "C91");

