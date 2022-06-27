#include "InitialConditions_analytical.h"
#include "AnalyticalFormula_tools.h"

namespace dyablo{

/**
 * 2D Orszag-Tang vortex test
 * Based on Orszag-Tang 1979 "Small-scale structure of two-dimensional magnetohydrodynamic turbulence"
 * Journal of Fluid Mechanics, 1979. Vol 90 pp 128-143
 **/
struct AnalyticalFormula_OrszagTang : public AnalyticalFormula_base{
  
  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;

  AnalyticalFormula_OrszagTang( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8))
  {
    assert(ndim == 2);
  }


  KOKKOS_INLINE_FUNCTION
  bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const 
  {
    const real_t gamma0 = this->gamma0;
    const real_t smallr = this->smallr;
    const real_t smallp = this->smallp;
    const real_t error_max = this->error_max;
    return AnalyticalFormula_tools::auto_refine( *this, gamma0, smallr, smallp, error_max,
                                                  x, y, z, dx, dy, dz );
  }

  KOKKOS_INLINE_FUNCTION
  ConsMHDState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    PrimMHDState res;
    constexpr real_t rho0 = 25.0 / (36.0*M_PI);
    constexpr real_t P0   = 5.0  / (12.0*M_PI);
    const real_t B0       = 1.0  / sqrt(4.0*M_PI);
    const real_t u        = -sin(2.0*M_PI*y);
    const real_t v        =  sin(2.0*M_PI*x);
    const real_t Bx       = -B0 * sin(2.0*M_PI*y);
    const real_t By       =  B0 * sin(4.0*M_PI*x);

    res.rho = rho0;
    res.u   = u;
    res.v   = v;
    res.w   = 0.0;
    res.p   = P0;
    res.Bx  = Bx;
    res.By  = By;
    res.Bz  = 0.0;
    
    ConsMHDState cons_res;
    cons_res = primToCons<3>(res, gamma0);
    return cons_res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_OrszagTang>, 
                 "orszag_tang");

