#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_hydro.hpp"

namespace dyablo {

/**
 * @brief Isentropic vortex model based on Spiegel et al. 2015
 */
struct AnalyticalFormula_isentropic_vortex : public AnalyticalFormula_base_hydro
{  
  const real_t alpha;    // Advection angle (in rad)
  const real_t gamma0;   // Adiabatic index
  const real_t Mach;     // Mach number
  const real_t R;        // Characteristic length scale
  const real_t sigma;    // Standard deviation for the Gaussian
  const real_t beta;     // Amplitude of the Gaussian

  // Constructor
  AnalyticalFormula_isentropic_vortex(ConfigMap &configMap) :
    alpha(configMap.getValue<real_t>("isentropic_vortex", "alpha_deg", 45.0) * M_PI / 180.0),
    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
    Mach(configMap.getValue<real_t>("isentropic_vortex", "Mach", Kokkos::sqrt(2.0 / gamma0))),
    R(configMap.getValue<real_t>("isentropic_vortex", "R", 1.0)),
    sigma(configMap.getValue<real_t>("isentropic_vortex", "sigma", 1.0)),
    beta(configMap.getValue<real_t>("isentropic_vortex", "beta", Mach * 5.0 * Kokkos::sqrt(2.0) / (4.0*M_PI) * Kokkos::exp(0.5))) 
  {}

  // Method called on each cell of the domain, potentially multiple times
  KOKKOS_INLINE_FUNCTION
  State value(real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz) const
  {
    State result{};

    // Perturbation strength (eq 21)
    const real_t R2 = R*R;
    const real_t f = -1.0 / (2.0*sigma*sigma) * (x*x + y*y) / R2;
    
    // Gaussian (eq 20)
    const real_t omega = beta * Kokkos::exp(f);

    // Initial perturbations (eq 22)
    const real_t du = -y/R * omega;
    const real_t dv =  x/R * omega;
    const real_t dT = -0.5*(gamma0-1.0) * omega*omega;

    // Primitive values (eq 23)
    const real_t rho = Kokkos::pow(1.0 + dT, 1.0 / (gamma0-1.0));
    const real_t u  = Mach * Kokkos::cos(alpha) + du;
    const real_t v  = Mach * Kokkos::sin(alpha) + dv;
    const real_t p  = Kokkos::pow(1.0 + dT, gamma0 / (gamma0-1.0)) / gamma0;

    // Kinetic energy
    const real_t Ek = 0.5 * rho * (u*u + v*v);
    const real_t e_tot = Ek + p / (gamma0-1.0);

    // Filling in State variable
    result.rho   = rho;
    result.rho_u = rho * u;
    result.rho_v = rho * v;
    result.rho_w = 0.0;
    result.e_tot = e_tot;

    return result;
  }
};
} // Namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_isentropic_vortex>, 
                 "isentropic_vortex");

