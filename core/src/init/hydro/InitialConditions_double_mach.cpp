#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_hydro.hpp"

namespace dyablo{

/**
 * Double Mach reflection
 * Based on Vevek 2019: https://dr.ntu.edu.sg/bitstream/10356/81953/1/On%20alternative%20setups%20of%20the%20double%20Mach%20reflection%20problem.pdf
 **/
struct AnalyticalFormula_double_mach : public AnalyticalFormula_base_hydro
{
  const real_t gamma0;

  const real_t x0;                         // Initial position of the interface
  const real_t post_rho, post_vel, post_p; // Initial value post-shock
  const real_t pre_rho, pre_vel, pre_p;    // Initial value pre-shock

  AnalyticalFormula_double_mach( ConfigMap& configMap ) : 
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.66666667)),
    x0(configMap.getValue<real_t>("double_mach", "x0", 0.1666667)),
    post_rho(configMap.getValue<real_t>("double_mach", "rho_left", 8.0)),
    post_vel(configMap.getValue<real_t>("double_mach", "vel_left", 8.25)),
    post_p(configMap.getValue<real_t>("double_mach", "p_left", 116.5)),
    pre_rho(configMap.getValue<real_t>("double_mach", "rho_right", 1.4)),
    pre_vel(configMap.getValue<real_t>("double_mach", "vel_right", 0.0)),
    pre_p(configMap.getValue<real_t>("double_mach", "p_right", 1.0))
  {}

  // KOKKOS_INLINE_FUNCTION
  // bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const 
  // {
  //   real_t xs = x0 + y / tan(alpha);
  //   return x-0.5*dx < xs && x+0.5*dx > xs;
  // }

  KOKKOS_INLINE_FUNCTION
  ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    const real_t alpha = M_PI/3.0;
    bool post_shock = x < (x0 + y / tan(alpha));

    ConsHydroState res{};
    if (post_shock) {
      res.rho   = post_rho;
      res.e_tot = 0.5 * (post_rho*post_vel*post_vel) + post_p / (gamma0-1.0);
      res.rho_u =  post_rho*post_vel*sin(alpha);
      res.rho_v = -post_rho*post_vel*cos(alpha);
    }
    else {
      res.rho   = pre_rho;
      res.rho_u =  pre_rho*pre_vel*sin(alpha);
      res.rho_v = -pre_rho*pre_vel*cos(alpha);
      res.e_tot = 0.5 * (pre_rho*pre_vel*pre_vel) + pre_p / (gamma0-1.0);
    }

    return res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_double_mach>, 
                 "double_mach");

