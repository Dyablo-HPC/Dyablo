#include "../InitialConditions_analytical.h"
#include "AnalyticalFormula_base_hydro.hpp"

namespace dyablo{

/**
 * 2D implosion test.
 * Based on Liska, Wendroff "Comparison of Several Difference Schemes on 1D and 2D Test Problems 
 * for the Euler Equations", 2003, SIAM journal on Scientific Computing
 **/
struct AnalyticalFormula_sod : public AnalyticalFormula_base_hydro
{
  const real_t gamma0;

  // pressure left/right of interface
  const real_t pL;
  const real_t pR;
  // density left/right of interface
  const real_t rhoL;
  const real_t rhoR;
  // angle of the sod
  const real_t angle;

  AnalyticalFormula_sod( ConfigMap& configMap ) : 
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),

    pL(configMap.getValue<real_t>("sod", "pL", 1.0)),
    pR(configMap.getValue<real_t>("sod", "pR", 0.1)),
    rhoL(configMap.getValue<real_t>("sod", "rhoL", 1.0)),
    rhoR(configMap.getValue<real_t>("sod", "rhoR", 0.125)),
    angle(configMap.getValue<real_t>("sod", "angle", 0.0))
  {}


  // KOKKOS_INLINE_FUNCTION
  // bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const 
  // {
  //   // Geometric refine : close to interface
  //   return fabs(x - this->x0) < 1.1*dx;
  // }

  KOKKOS_INLINE_FUNCTION
  ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    real_t p, rho;
    if (angle == 0.0) {
      p = (x < 0.5 ? pL : pR);
      rho = (x < 0.5 ? rhoL : rhoR);
    }
    else {
      // Finding the x0 so that the separation goes through the middle of the domain
      const real_t Rdx = Kokkos::sin(angle);
      const real_t Rdy = Kokkos::cos(angle);
      const real_t k = 0.5 / Rdy;
      const real_t x0 = 0.5 + k * Rdx; //0.5 - 0.5*Rdx/Rdy;

      printf("Rdx=%lf; Rdy=%lf; k=%lf; x0=%lf\n", Rdx, Rdy, k, x0);

      const real_t ytheta = (x0-x) * Rdy/Rdx;
      p = (y < ytheta) ? pL : pR;
      rho = (y < ytheta) ? rhoL : rhoR;
    }

    ConsHydroState res{};
    res.rho   = rho;
    res.e_tot = p / (gamma0-1.0);

    return res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_sod>, 
                 "sod");

