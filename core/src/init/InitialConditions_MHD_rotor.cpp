#include "InitialConditions_analytical.h"
#include "AnalyticalFormula_tools.h"
#include "states/State_forward.h"

namespace dyablo{

template <typename State>
struct AnalyticalFormula_MHD_rotor : public AnalyticalFormula_base{
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;
  
  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;
  const real_t xmin, ymin;
  const real_t xmax, ymax;
  const real_t xmid, ymid;

  AnalyticalFormula_MHD_rotor( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
    ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
    xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
    ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
    xmid(0.5*(xmin+xmax)),
    ymid(0.5*(ymin+ymax))
    
  {
    DYABLO_ASSERT_HOST_RELEASE(ndim == 2, "Initial conditions only for 2D");
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
  ConsState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    PrimState res;
    constexpr real_t p0   = 1.0;
    constexpr real_t rho0 = 10.0;
    constexpr real_t rho2 = 1.0;
    constexpr real_t r0 = 0.1;
    constexpr real_t r1 = 0.115;
    constexpr real_t u0 = 2.0;

    const real_t ddx = xmid-x;
    const real_t ddy = ymid-y;

    const real_t r2 = ddx*ddx+ddy*ddy;
    const real_t r = sqrt(r2);
    const real_t f = (r1-r)/(r1-r0);
    const real_t q = u0/r0;
    
    real_t rho, u, v;
    if (r < r0) {
      rho = rho0;
      u = q*ddy;
      v = q*ddx;
    }
    else if (r > r1) {
      rho = rho2;
      u = 0.0;
      v = 0.0;
    }
    else {
      rho = rho2 + (rho0-rho2)*f;
      u = f*q*ddy;
      v = f*q*ddx;
    }

    const real_t Bx = 5.0 / sqrt(4.0*M_PI);

    res.rho = rho;
    res.u   = u;
    res.v   = v;
    res.w   = 0.0;
    res.p   = p0;
    res.Bx  = Bx;
    res.By  = 0.0;
    res.Bz  = 0.0;
    
    if constexpr (std::is_same_v<PrimState, PrimGLMMHDState>)
      res.psi = 0.0;

    ConsState cons_res;
    cons_res = primToCons<3>(res, gamma0);
    return cons_res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_rotor<dyablo::MHDState>>, 
                 "MHD_rotor");

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_rotor<dyablo::GLMMHDState>>, 
                 "MHD_rotor_glm");
