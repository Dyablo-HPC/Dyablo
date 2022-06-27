/**
 * \file RiemannSolvers.h
 * All possible Riemann solvers or so.
 */
#ifndef RIEMANN_SOLVERS_H_
#define RIEMANN_SOLVERS_H_

#include <math.h>

#include "states/State_forward.h"

namespace dyablo {

//! Riemann solver type for hydro fluxes
enum RiemannSolverType {
  RIEMANN_APPROX, /*!< quasi-exact Riemann solver (hydro-only) */ 
  RIEMANN_LLF,    /*!< LLF Local Lax-Friedrich */
  RIEMANN_HLL,    /*!< HLL hydro and MHD Riemann solver */
  RIEMANN_HLLC,   /*!< HLLC hydro-only Riemann solver */
  RIEMANN_MHD     /*!< MHD only solver */
  //RIEMANN_HLLD    /*!< HLLD MHD-only Riemann solver */
};

} // namespace dyablo

template<>
inline named_enum<dyablo::RiemannSolverType>::init_list named_enum<dyablo::RiemannSolverType>::names = {
    {dyablo::RiemannSolverType::RIEMANN_APPROX, "approx"},
    {dyablo::RiemannSolverType::RIEMANN_LLF, "llf"},
    {dyablo::RiemannSolverType::RIEMANN_HLL, "hll"},
    {dyablo::RiemannSolverType::RIEMANN_HLLC, "hllc"},
    {dyablo::RiemannSolverType::RIEMANN_MHD,  "five_waves"}
};

namespace dyablo {

struct RiemannParams
{
  RiemannParams( ConfigMap& configMap )
  : riemannSolverType( configMap.getValue<RiemannSolverType>("hydro","riemann", RIEMANN_HLLC)),
    gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
    gamma6( (gamma0 + 1) / (2*gamma0)),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallp( configMap.getValue<real_t>("hydro","smallp", 1e-10) ),
    smallc( sqrt(smallp*gamma0/smallr) ),
    smallpp( smallr*smallp ),
    three_waves( configMap.getValue<bool>("hydro", "three_waves", false))
  { }

  RiemannSolverType riemannSolverType;

  real_t gamma0;
  real_t gamma6;
  real_t smallr;
  real_t smallp;
  real_t smallc;
  real_t smallpp;

  bool three_waves;
};  

/** 
 * Riemann solver HLLC
 *
 * @param[in] qleft  : input left  state (primitive variables)
 * @param[in] qright : input right state (primitive variables)
 * @param[out] flux  : output flux
 */
KOKKOS_INLINE_FUNCTION
void riemann_hllc(const PrimHydroState& qleft,
		              const PrimHydroState& qright,
		              ConsHydroState& flux,
		              const RiemannParams& params)
{
  real_t gamma0 = params.gamma0;
  real_t smallr = params.smallr;
  real_t smallp = params.smallp;
  real_t smallc = params.smallc;
  
  const real_t entho = ONE_F / (gamma0 - ONE_F);
  
  // Left variables
  real_t rl = fmax(qleft.rho, smallr);
  real_t pl = fmax(qleft.p, rl*smallp);
  real_t ul =      qleft.u;
  real_t vl =      qleft.v;
  real_t wl =      qleft.w;
    
  real_t ecinl = HALF_F*rl*(ul*ul+vl*vl+wl*wl);
  real_t etotl = pl*entho+ecinl;
  real_t ptotl = pl;

  // Right variables
  real_t rr = fmax(qright.rho, smallr);
  real_t pr = fmax(qright.p, rr*smallp);
  real_t ur =      qright.u;
  real_t vr =      qright.v;
  real_t wr =      qright.w;

  real_t ecinr = HALF_F*rr*(ur*ur+vr*vr+wr*wr);
  real_t etotr = pr*entho+ecinr;
  real_t ptotr = pr;
    
  // Find the largest eigenvalues in the normal direction to the interface
  real_t cfastl = SQRT(fmax(gamma0*pl/rl,smallc*smallc));
  real_t cfastr = SQRT(fmax(gamma0*pr/rr,smallc*smallc));

  // Compute HLL wave speed
  real_t SL = fmin(ul,ur) - fmax(cfastl,cfastr);
  real_t SR = fmax(ul,ur) + fmax(cfastl,cfastr);

  // Compute lagrangian sound speed
  real_t rcl = rl*(ul-SL);
  real_t rcr = rr*(SR-ur);
    
  // Compute acoustic star state
  real_t ustar    = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
  real_t ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

  // Left star region variables
  real_t rstarl    = rl*(SL-ul)/(SL-ustar);
  real_t etotstarl = ((SL-ul)*etotl-ptotl*ul+ptotstar*ustar)/(SL-ustar);
    
  // Right star region variables
  real_t rstarr    = rr*(SR-ur)/(SR-ustar);
  real_t etotstarr = ((SR-ur)*etotr-ptotr*ur+ptotstar*ustar)/(SR-ustar);
    
  // Sample the solution at x/t=0
  real_t ro, uo, ptoto, etoto;
  if (SL > ZERO_F) {
    ro=rl;
    uo=ul;
    ptoto=ptotl;
    etoto=etotl;
  } else if (ustar > ZERO_F) {
    ro=rstarl;
    uo=ustar;
    ptoto=ptotstar;
    etoto=etotstarl;
  } else if (SR > ZERO_F) {
    ro=rstarr;
    uo=ustar;
    ptoto=ptotstar;
    etoto=etotstarr;
  } else {
    ro=rr;
    uo=ur;
    ptoto=ptotr;
    etoto=etotr;
  }
      
  // Compute the Godunov flux
  flux.rho   = ro*uo;
  flux.rho_u = ro*uo*uo+ptoto;
  flux.e_tot = (etoto+ptoto)*uo;
  if (flux.rho > ZERO_F) {
    flux.rho_v = flux.rho*qleft.v;
    flux.rho_w = flux.rho*qleft.w;
  } else {
    flux.rho_v = flux.rho*qright.v;
    flux.rho_w = flux.rho*qright.w;
  }
} // riemann_hllc

KOKKOS_INLINE_FUNCTION
void riemann_mhd(const PrimMHDState& qleft, 
                 const PrimMHDState& qright, 
                       ConsMHDState& flux, 
                 const RiemannParams& params) {
  using vec_t = StateNd<3>;

  constexpr real_t epsilon = 1.0e-16; // No std::numeric_limits

  // Left quantities
  const real_t emagL = 0.5 * (qleft.Bx*qleft.Bx + qleft.By*qleft.By + qleft.Bz*qleft.Bz); 
  const real_t B2L   = qleft.Bx*qleft.Bx;
  const real_t B2TL  = 2.0*emagL - B2L;
  const real_t BNBTL = sqrt(B2L*B2TL);   
  const real_t cs_L  = sqrt(params.gamma0 * qleft.p / qleft.rho);
        real_t c_AL  = sqrt(qleft.rho * (1.5*B2L + 0.5*B2TL)) + epsilon;
        real_t c_BL  = sqrt(qleft.rho*(qleft.rho*cs_L*cs_L + 1.5*B2TL + 0.5*B2L));


  // Right quantities 
  const real_t emagR = 0.5 * (qright.Bx*qright.Bx + qright.By*qright.By + qright.Bz*qright.Bz);    
  const real_t B2R   = qright.Bx*qright.Bx;
  const real_t B2TR  = 2.0*emagR - B2R;
  const real_t BNBTR = sqrt(B2R*B2TR);   
  const real_t cs_R  = sqrt(params.gamma0 * qright.p / qright.rho);
        real_t c_AR  = sqrt(qright.rho * (1.5*B2R + 0.5*B2TR)) + epsilon;
        real_t c_BR  = sqrt(qright.rho*(qright.rho*cs_R*cs_R + 1.5*B2TR + 0.5*B2R));

  const vec_t pL {-qleft.Bx * qleft.Bx + emagL + qleft.p,
                  -qleft.Bx * qleft.By,
                  -qleft.Bx * qleft.Bz};
  const vec_t pR {-qright.Bx * qright.Bx + emagR + qright.p,
                  -qright.Bx * qright.By,
                  -qright.Bx * qright.Bz};

  auto computeFastMagnetoAcousticSpeed = [&](const PrimMHDState &q, const real_t B2, const real_t cs) {
    const real_t c02  = cs*cs;
    const real_t ca2  = B2 / q.rho;
    const real_t cap2 = q.Bx*q.Bx/q.rho;
    // Remi's version
    return sqrt(0.5*(c02+ca2)+0.5*sqrt((c02+ca2)*(c02+ca2)-4.0*c02*cap2));
  };

  // Using 3-wave if hyperbolicity is lost
  if ( qleft.Bx*qright.Bx < -epsilon 
    || qleft.By*qright.By < -epsilon
    || qleft.Bz*qright.Bz < -epsilon 
    || params.three_waves) {
    
    const real_t cL = qleft.rho  * computeFastMagnetoAcousticSpeed(qleft,  emagL*2.0, cs_L);
    const real_t cR = qright.rho * computeFastMagnetoAcousticSpeed(qright, emagR*2.0, cs_R);
    const real_t c = fmax(cL, cR);

    c_AL = c;
    c_AR = c;
    c_BL = c;
    c_BR = c;
  }

  const real_t inv_sum_A = 1.0 / (c_AL+c_AR);
  const real_t inv_sum_B = 1.0 / (c_BL+c_BR);
  const vec_t cL {c_BL, c_AL, c_AL};
  const vec_t cR {c_BR, c_AR, c_AR};
  const vec_t clpcrm1 {inv_sum_B, inv_sum_A, inv_sum_A};
  
  const vec_t vL {qleft.u,  qleft.v,  qleft.w};
  const vec_t vR {qright.u, qright.v, qright.w};

  const vec_t ustar = clpcrm1 * (cL*vL + cR*vR + pL - pR);
  const vec_t pstar = clpcrm1 * (cR*pL + cL*pR + cL*cR*(vL-vR));

  PrimMHDState qr;
  real_t Bnext;
  if (ustar[IX] > 0.0) {
    qr = qleft;
    Bnext = qright.Bx;
  }
  else {
    qr = qright;
    Bnext = qleft.Bx;
  }

  ConsMHDState ur = primToCons<3>(qr, params.gamma0);
  real_t us = ustar[IX];
  flux.rho   = us*ur.rho;
  flux.rho_u = us*ur.rho_u + pstar[IX];
  flux.rho_v = us*ur.rho_v + pstar[IY];
  flux.rho_w = us*ur.rho_w + pstar[IZ];
  flux.e_tot = us*ur.e_tot + (pstar[IX]*ustar[IX]+pstar[IY]*ustar[IY]+pstar[IZ]*ustar[IZ]);
  flux.Bx    = us*ur.Bx - Bnext*ustar[IX];
  flux.By    = us*ur.By - Bnext*ustar[IY];
  flux.Bz    = us*ur.Bz - Bnext*ustar[IZ];
}

/**
 * Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
void riemann_hydro( const PrimHydroState& qleft,
		                const PrimHydroState& qright,
		                ConsHydroState& flux,
		                const RiemannParams& params)
{
  riemann_hllc(qleft,qright,flux,params);
} // riemann_hydro

/**
 * Another Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
ConsHydroState riemann_hydro( const PrimHydroState& qleft,
                              const PrimHydroState& qright,
                              const RiemannParams& params)
{
  ConsHydroState flux;
  riemann_hydro(qleft, qright, flux, params);
  return flux;

} // riemann_hydro

/**
 * Another Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
ConsMHDState riemann_hydro( const PrimMHDState& qleft,
                            const PrimMHDState& qright,
                            const RiemannParams& params)
{
  ConsMHDState flux;
  riemann_mhd(qleft, qright, flux, params);
  return flux;

} // riemann_hydro

} // namespace dyablo

#endif // RIEMANN_SOLVERS_H_
