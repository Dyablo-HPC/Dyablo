/**
 * \file RiemannSolvers.h
 * All possible Riemann solvers or so.
 */
#ifndef RIEMANN_SOLVERS_H_
#define RIEMANN_SOLVERS_H_

#include <math.h>
#include "states/State_Nd.h"
#include "states/State_forward.h"

namespace dyablo {

//! Riemann solver type for hydro fluxes
enum RiemannSolverType {
  RIEMANN_HLL,         /*!< HLL hydro and MHD Riemann solver */
  RIEMANN_HLLC,       /*!< HLLC hydro-only Riemann solver */
  RIEMANN_FIVE_WAVES, /*!< MHD only solver */
  RIEMANN_HLLD        /*!< HLLD MHD-only Riemann solver */
};

} // namespace dyablo

template<>
inline named_enum<dyablo::RiemannSolverType>::init_list named_enum<dyablo::RiemannSolverType>::names()
{
  return{
    {dyablo::RiemannSolverType::RIEMANN_HLLC,       "hllc"},
    {dyablo::RiemannSolverType::RIEMANN_HLLD,       "hlld"},
    {dyablo::RiemannSolverType::RIEMANN_FIVE_WAVES, "five_waves"}
  };
}

namespace dyablo {

struct RiemannParams
{
  RiemannParams( ConfigMap& configMap )
  : riemannSolverType( configMap.getValue<RiemannSolverType>("hydro","riemann", RIEMANN_HLLC)),
    gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
    gamma6( (gamma0 + 1) / (2*gamma0)),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallp( configMap.getValue<real_t>("hydro","smallp", 1e-10) ),
    smalle( configMap.getValue<real_t>("hydro","smalle", 1e-5) ),
    smallc( sqrt(smallp*gamma0/smallr) ),
    smallpp( smallr*smallp ),
    three_waves( configMap.getValue<bool>("hydro", "three_waves", false))
  { 
    const bool use_glm = configMap.getValue<std::string>("hydro", "update", "HydroUpdate_hancock").find("GLMMHD") != std::string::npos;
    if (use_glm) {
      const uint32_t ndim = configMap.getValue<uint32_t>("mesh", "ndim", 3);
      const real_t xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0);
      const real_t xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0);
      const real_t ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0);
      const real_t ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0);
      const real_t zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0);
      const real_t zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0);

      const uint32_t level_max = configMap.getValue<uint32_t>("amr", "level_max", 10);

      const uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 0);
      const uint32_t by = configMap.getValue<uint32_t>("amr", "by", 0);
      const uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", 1);

      const real_t Lx = xmax - xmin;
      const real_t Ly = ymax - ymin;
      const real_t Lz = zmax - zmin; 

      const real_t min_dx = Lx / ((1 << level_max) * bx);
      const real_t min_dy = Ly / ((1 << level_max) * by);
      const real_t min_dz = Lz / ((1 << level_max) * bz); 

      real_t min_dh = FMIN(min_dx, min_dy);
      if (ndim == 3)
        min_dh = FMIN(min_dh, min_dz);

      const real_t cfl = configMap.getValue<real_t>("hydro", "cfl", 0.5);

      ch = 0.5*cfl * min_dh; // This value must be divided by dt in the kernels !!!
      cr = configMap.getValue<real_t>("hydro", "cr", 0.1);
    }

  }

  RiemannSolverType riemannSolverType;

  real_t gamma0;
  real_t gamma6;
  real_t smallr;
  real_t smallp;
  real_t smalle;
  real_t smallc;
  real_t smallpp;

  bool three_waves;
  real_t ch, cr;
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
		              const RiemannParams& params,
                  real_t &p_out)
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
    p_out=pl;
  } else if (ustar > ZERO_F) {
    ro=rstarl;
    uo=ustar;
    ptoto=ptotstar;
    etoto=etotstarl;
    p_out=ptotstar;
  } else if (SR > ZERO_F) {
    ro=rstarr;
    uo=ustar;
    ptoto=ptotstar;
    etoto=etotstarr;
    p_out=ptotstar;
  } else {
    ro=rr;
    uo=ur;
    ptoto=ptotr;
    etoto=etotr;
    p_out=pr;
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
void riemann_five_waves(const PrimMHDState&  qleft, 
                        const PrimMHDState&  qright, 
                              ConsMHDState&  flux,
                              real_t&        p_out, 
                        const RiemannParams& params) {
  using vec_t = StateNd<3>;

  constexpr real_t epsilon = 1.0e-16; // No std::numeric_limits

  // Left quantities
  const real_t emagL = 0.5 * (qleft.Bx*qleft.Bx + qleft.By*qleft.By + qleft.Bz*qleft.Bz); 
  const real_t B2L   = qleft.Bx*qleft.Bx;
  const real_t B2TL  = 2.0*emagL - B2L;
  //const real_t BNBTL = sqrt(B2L*B2TL);   
  const real_t cs_L  = sqrt(params.gamma0 * qleft.p / qleft.rho);
        real_t c_AL  = sqrt(qleft.rho * (1.5*B2L + 0.5*B2TL)) + epsilon;
        real_t c_BL  = sqrt(qleft.rho*(qleft.rho*cs_L*cs_L + 1.5*B2TL + 0.5*B2L));


  // Right quantities 
  const real_t emagR = 0.5 * (qright.Bx*qright.Bx + qright.By*qright.By + qright.Bz*qright.Bz);    
  const real_t B2R   = qright.Bx*qright.Bx;
  const real_t B2TR  = 2.0*emagR - B2R;
  //const real_t BNBTR = sqrt(B2R*B2TR);   
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

  PrimMHDState qr{};
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

  p_out = pstar[IX];
}


/** 
 * Riemann solver HLLD (MHD)
 * 
 * Refs:
 *  . Miyoshi, T., Kusano, K., "A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics", Journal of Computational Physics, 2005
 *
 * @param[in] qleft  : input left  state (primitive variables)
 * @param[in] qright : input right state (primitive variables)
 * @param[out] flux  : output flux
 * @param[in] params : parameters of the Riemann solver and physics
 * @param[in] Bx : estimated value of Bx at the interface
 */
KOKKOS_INLINE_FUNCTION
void riemann_hlld( const PrimMHDState&  qleft, 
                   const PrimMHDState&  qright, 
                         ConsMHDState&  flux,
                         real_t& p_out,
                   const RiemannParams& params,
                   const real_t Bx)
{
  const real_t Bsgn = (Bx < 0.0 ? -1.0 : 1.0);

  // Deref of states
  const real_t uL = qleft.u;
  const real_t vL = qleft.v;
  const real_t wL = qleft.w;
  const real_t pL = qleft.p;
  const real_t rL = qleft.rho;
  const real_t ByL = qleft.By;
  const real_t BzL = qleft.Bz;
  const real_t B2L = Bx*Bx+ByL*ByL+BzL*BzL;
  const real_t pTL = pL + 0.5 * B2L;
  const real_t EL  = pL / (params.gamma0-1.0) + 0.5*rL*(uL*uL+vL*vL+wL*wL) + 0.5*B2L;

  const real_t uR = qright.u;
  const real_t vR = qright.v;
  const real_t wR = qright.w;
  const real_t pR = qright.p;
  const real_t rR = qright.rho;
  const real_t ByR = qright.By;
  const real_t BzR = qright.Bz;
  const real_t B2R = Bx*Bx+ByR*ByR+BzR*BzR;
  const real_t pTR = pR + 0.5 * B2R;
  const real_t ER  = pR / (params.gamma0-1.0) + 0.5*rR*(uR*uR+vR*vR+wR*wR) + 0.5*B2R;


  // Calculating left and right fast-magnetosonic waves
  //
  /* TODO : Check that this is actually the same as in five_waves and maybe extract this to 
     State_MHD.h ?
  */
  auto computeFastMagnetoAcousticSpeed = [&](const PrimMHDState &q) {
    const real_t gp = params.gamma0 * q.p;
    const real_t B2 = Bx*Bx + q.By*q.By + q.Bz*q.Bz;
    
    return sqrt(0.5 * (gp + B2 + sqrt((gp + B2)*(gp + B2) - 4.0*gp*Bx*Bx)) / q.rho);
  };

  const real_t cfL = computeFastMagnetoAcousticSpeed(qleft);
  const real_t cfR = computeFastMagnetoAcousticSpeed(qright);
  
  // HLL Wave speed
  const real_t SL = fmin(uL, uR) - fmax(cfL, cfR);
  const real_t SR = fmax(uL, uR) + fmax(cfL, cfR);

  // Lagrangian speed of sound
  const real_t rCL = rL*(uL-SL);
  const real_t rCR = rR*(SR-uR);

  // Entropy wave speed
  const real_t uS = (rCR*uR + rCL*uL - pTR + pTL) / (rCR+rCL);
  
  // Single Star state
  const real_t pTS = (rCR*pTL + rCL*pTR - rCR*rCL*(uR-uL)) / (rCR+rCL); 

  // Single star densities
  const real_t rLS = rL * (SL-uL)/(SL-uS);
  const real_t rRS = rR * (SR-uR)/(SR-uS);

  // Single star velocities
  const real_t econvL = rL*(SL-uL)*(SL-uS)-Bx*Bx;
  const real_t econvR = rR*(SR-uR)*(SR-uS)-Bx*Bx;

  const real_t uconvL = (uS-uL) / econvL;
  const real_t uconvR = (uS-uR) / econvR;
  const real_t BconvL = (rCL*rCL/rL - Bx*Bx) / econvL;
  const real_t BconvR = (rCR*rCR/rR - Bx*Bx) / econvR;

  real_t vLS, vRS, wLS, wRS, ByLS, ByRS, BzLS, BzRS;

  // Switching to two state on the left ?
  if (FABS(econvL) < params.smalle*Bx*Bx) {
    vLS = vL;
    wLS = wL;
    ByLS = ByL;
    BzLS = BzL;
  }
  else {
    vLS = vL - Bx*ByL * uconvL;
    wLS = wL - Bx*BzL * uconvL;
    ByLS = ByL * BconvL;
    BzLS = BzL * BconvL;
  }

  // Switching to two state on the right ?
  if (FABS(econvR) < params.smalle*Bx*Bx) {
    vRS  = vR;
    wRS  = wR;
    ByRS = ByR;
    BzRS = BzR;
  }
  else {
    vRS = vR - Bx*ByR * uconvR;
    wRS = wR - Bx*BzR * uconvR;
    ByRS = ByR * BconvR;
    BzRS = BzR * BconvR;
  }

  // Single star total energy
  const real_t udotBL = uL*Bx+vL*ByL+wL*BzL;
  const real_t udotBR = uR*Bx+vR*ByR+wR*BzR;
  const real_t uSdotBSL = uS*Bx+vLS*ByLS+wLS*BzLS;
  const real_t uSdotBSR = uS*Bx+vRS*ByRS+wRS*BzRS;

  const real_t ELS = ((SL-uL)*EL - pTL*uL + pTS*uS + Bx*(udotBL - uSdotBSL)) / (SL-uS);
  const real_t ERS = ((SR-uR)*ER - pTR*uR + pTS*uS + Bx*(udotBR - uSdotBSR)) / (SR-uS);

  // Alfven wave speeds
  const real_t srLS = sqrt(rLS);
  const real_t srRS = sqrt(rRS);
  const real_t SLS = uS - fabs(Bx) / srLS;
  const real_t SRS = uS + fabs(Bx) / srRS;

  // Double Star state
  const real_t den_fac = 1.0 / (srLS + srRS);
  const real_t vSS = (srLS*vLS + srRS*vRS + (ByRS-ByLS)*Bsgn) * den_fac;
  const real_t wSS = (srLS*wLS + srRS*wRS + (BzRS-BzLS)*Bsgn) * den_fac;
  const real_t BySS = (srLS*ByRS + srRS*ByLS + srLS*srRS*(vRS-vLS)*Bsgn) * den_fac;
  const real_t BzSS = (srLS*BzRS + srRS*BzLS + srLS*srRS*(wRS-wLS)*Bsgn) * den_fac; 

  const real_t uSSdotBSS = uS*Bx + vSS*BySS + wSS*BzSS;

  const real_t ELSS = ELS - srLS * (uSdotBSL - uSSdotBSS) * Bsgn;
  const real_t ERSS = ERS + srRS * (uSdotBSR - uSSdotBSS) * Bsgn;

  // Lambda to compute a flux from a primitive state
  auto computeFlux = [&](const PrimMHDState &q, const real_t e_tot) -> ConsMHDState {
    ConsMHDState res{};

    res.rho   = q.rho * q.u;
    res.rho_u = q.rho * q.u * q.u + q.p - q.Bx*q.Bx;
    res.rho_v = q.rho * q.u * q.v - q.Bx*q.By;
    res.rho_w = q.rho * q.u * q.w - q.Bx*q.Bz;
    res.Bx    = 0.0;
    res.By    = q.By*q.u - q.Bx*q.v;
    res.Bz    = q.Bz*q.u - q.Bx*q.w;
    res.e_tot = (e_tot + q.p) * q.u - q.Bx*(q.Bx*q.u+q.By*q.v+q.Bz*q.w);
    
    return res;
  };

  // Disjunction of cases
  
  PrimMHDState q;
  real_t e_tot;
  if (SL > 0.0) { // qL
    q = qleft;
    e_tot = EL;
    q.p = pTL;

    p_out = qleft.p;
  }
  else if (SLS > 0.0) { // qL*
    q.rho = rLS;
    q.u   = uS;
    q.v   = vLS;
    q.w   = wLS;
    q.Bx  = Bx;
    q.By  = ByLS;
    q.Bz  = BzLS;

    q.p = pTS;
    e_tot = ELS;

    p_out = qleft.p;
  }
  else if (uS > 0.0) { // qL**
    q.rho = rLS;
    q.u   = uS;
    q.v   = vSS;
    q.w   = wSS;
    q.Bx  = Bx;
    q.By  = BySS;
    q.Bz  = BzSS;

    q.p   = pTS;
    e_tot = ELSS;

    p_out = qleft.p;
  }
  else if (SRS > 0.0) { // qR**
    q.rho = rRS;
    q.u   = uS;
    q.v   = vSS;
    q.w   = wSS;
    q.Bx  = Bx;
    q.By  = BySS;
    q.Bz  = BzSS;

    q.p   = pTS;
    e_tot = ERSS;

    p_out = qright.p;
  }
  else if (SR > 0.0) { // qR*
    q.rho = rRS;
    q.u   = uS;
    q.v   = vRS;
    q.w   = wRS;
    q.Bx  = Bx;
    q.By  = ByRS;
    q.Bz  = BzRS;

    q.p = pTS;
    e_tot = ERS;

    p_out = qright.p;
  }
  else { // SR < 0.0; qR
    q = qright;
    e_tot = ER;
    q.p = pTR;

    p_out = qright.p;
  }

  flux = computeFlux(q, e_tot);
}


/**
 * Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
void riemann_hydro( const PrimHydroState& qleft,
		                const PrimHydroState& qright,
		                ConsHydroState& flux,
		                const RiemannParams& params,
                    real_t &p_out)
{
  riemann_hllc(qleft,qright,flux,params,p_out);
} // riemann_hydro

KOKKOS_INLINE_FUNCTION
void riemann_hydro( const PrimHydroState& qleft,
		                const PrimHydroState& qright,
		                ConsHydroState& flux,
		                const RiemannParams& params)
{
  real_t p_out; // Discarded
  riemann_hllc(qleft,qright,flux,params,p_out);
} // riemann_hydro

/**
 * Another Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
ConsHydroState riemann_hydro( const PrimHydroState& qleft,
                              const PrimHydroState& qright,
                              const RiemannParams& params,
                              real_t &p_out,
                              real_t dt)
{
  ConsHydroState flux;
  riemann_hydro(qleft, qright, flux, params, p_out);
  return flux;

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
  real_t p_out; // Discarded
  riemann_hydro(qleft, qright, flux, params, p_out);
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
  ConsMHDState flux{};
  real_t p_out; // Discarded
  riemann_five_waves(qleft, qright, flux, p_out, params);
  return flux;

} // riemann_hydro

/**
 * Another Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
ConsMHDState riemann_hydro( const PrimMHDState& qleft,
                            const PrimMHDState& qright,
                            const RiemannParams& params,
                            real_t &p_out,
                            real_t dt)
{
  ConsMHDState flux{};
  riemann_five_waves(qleft, qright, flux, p_out, params);
  return flux;

} // riemann_hydro

/**
 * @brief Riemann scheme with GLM correction for div.B
 * 
 * @param qleft Left reconstructed state of the interface
 * @param qright Right reconstructed state of the interface
 * @param params Riemann parameters
 * @param left_is_center Is the current cell being updated the left one ?
 * @param dt Time step
 * @return The conservative flux for the GLM-MHD equations 
 */
KOKKOS_INLINE_FUNCTION
ConsGLMMHDState riemann_hydro( const PrimGLMMHDState& qleft,
                               const PrimGLMMHDState& qright,
                               const RiemannParams&   params,
                                     real_t& p_out,
                               const real_t  dt) {
  ConsGLMMHDState flux{};
  ConsMHDState flux_mhd{};
  PrimMHDState qleft_mhd  = GLMToMHD(qleft);
  PrimMHDState qright_mhd = GLMToMHD(qright);

  const real_t ch = params.ch / dt;
  const real_t cp = sqrt(params.cr*ch);
  const real_t parabolic_term = exp(-0.5 * dt * ch*ch/(cp*cp));

  // Parabolic update
  const real_t psi_left  = qleft.psi * parabolic_term;
  const real_t psi_right = qright.psi * parabolic_term;

  // Applying traditional hlld solver on flux
  const real_t Bx_m = qleft.Bx + 0.5 * (qright.Bx - qleft.Bx) - 0.5/ch * (psi_right - psi_left);
  if (params.riemannSolverType == RiemannSolverType::RIEMANN_FIVE_WAVES)
    riemann_five_waves(qleft_mhd, qright_mhd, flux_mhd, p_out, params);
  else if (params.riemannSolverType == RiemannSolverType::RIEMANN_HLLD)
    riemann_hlld(qleft_mhd, qright_mhd, flux_mhd, p_out, params, Bx_m);
  else
    assert(false);

  // Applying flux correction to hyperbolic cleaning
  const real_t psi_m = psi_left + 0.5 * (psi_right - psi_left) - 0.5*ch * (qright.Bx - qleft.Bx);
  real_t flux_Bx  = psi_m;
  real_t flux_psi = ch*ch*Bx_m; 

  flux.rho   = flux_mhd.rho;
  flux.rho_u = flux_mhd.rho_u;
  flux.rho_v = flux_mhd.rho_v;
  flux.rho_w = flux_mhd.rho_w;
  flux.e_tot = flux_mhd.e_tot;
  flux.Bx    = flux_mhd.Bx + flux_Bx;
  flux.By    = flux_mhd.By;
  flux.Bz    = flux_mhd.Bz;
  flux.psi   = flux_psi;

  return flux;
}

} // namespace dyablo

#endif // RIEMANN_SOLVERS_H_
