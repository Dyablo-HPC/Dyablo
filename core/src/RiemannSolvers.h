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
  //RIEMANN_HLLD    /*!< HLLD MHD-only Riemann solver */
};

} // namespace dyablo

template<>
inline named_enum<dyablo::RiemannSolverType>::init_list named_enum<dyablo::RiemannSolverType>::names = {
    {dyablo::RiemannSolverType::RIEMANN_APPROX, "approx"},
    {dyablo::RiemannSolverType::RIEMANN_LLF, "llf"},
    {dyablo::RiemannSolverType::RIEMANN_HLL, "hll"},
    {dyablo::RiemannSolverType::RIEMANN_HLLC, "hllc"},
};

namespace dyablo {

struct RiemannParams
{
  RiemannParams( ConfigMap& configMap )
  : riemannSolverType( configMap.getValue<RiemannSolverType>("hydro","riemann", RIEMANN_HLLC)),
    gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
    gamma6( (gamma0 + 1) / (2*gamma0)),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallc( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
    smallp( smallc*smallc/gamma0 ),
    smallpp( smallr*smallp )
  { }

  RiemannSolverType riemannSolverType;

  real_t gamma0;
  real_t gamma6;
  real_t smallr;
  real_t smallc;
  real_t smallp;
  real_t smallpp;
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
 * TEMPORARY !!!!!
 * Wrapper function calling the actual riemann solver.
 */
KOKKOS_INLINE_FUNCTION
void riemann_hydro( const PrimMHDState& qleft,
		                const PrimMHDState& qright,
		                ConsMHDState& flux,
		                const RiemannParams& params)
{

  PrimHydroState qleft_{qleft.rho, qleft.p, qleft.u, qleft.v, qleft.w};
  PrimHydroState qright_{qright.rho, qright.p, qright.u, qright.v, qright.w};
  ConsHydroState flux_;
  riemann_hllc(qleft_,qright_,flux_,params);
  flux = {flux_.rho, flux_.e_tot, flux_.rho_u, flux_.rho_v, flux_.rho_w};
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
  riemann_hydro(qleft, qright, flux, params);
  return flux;

} // riemann_hydro

} // namespace dyablo

#endif // RIEMANN_SOLVERS_H_
