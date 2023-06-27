#include "InitialConditions_analytical.h"

#include "AnalyticalFormula_tools.h"

namespace dyablo{

/**
 * 2D implosion test.
 * Based on Liska, Wendroff "Comparison of Several Difference Schemes on 1D and 2D Test Problems 
 * for the Euler Equations", 2003, SIAM journal on Scientific Computing
 **/
struct AnalyticalFormula_sod : public AnalyticalFormula_base{
  const real_t gamma0;

  const real_t x0; // Initial position of the interface
  // pressure left/right of interface
  const real_t pL;
  const real_t pR;
  // density left/right of interface
  const real_t rhoL;
  const real_t rhoR;

  AnalyticalFormula_sod( ConfigMap& configMap ) : 
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),

    x0(configMap.getValue<real_t>("sod", "x0", 0.5)),
    pL(configMap.getValue<real_t>("sod", "pL", 1.0)),
    pR(configMap.getValue<real_t>("sod", "pR", 0.1)),
    rhoL(configMap.getValue<real_t>("sod", "rhoL", 1.0)),
    rhoR(configMap.getValue<real_t>("sod", "rhoR", 0.125))
  {}


  KOKKOS_INLINE_FUNCTION
  bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const 
  {
    // Geometric refine : close to interface
    return fabs(x - this->x0) < 1.1*dx;
  }

  KOKKOS_INLINE_FUNCTION
  ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    real_t p = (x < this->x0) ? pL : pR;
    real_t rho = (x < this->x0) ? rhoL : rhoR;

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

