/**
 * \file utils_hydro.h
 * \author Pierre Kestener
 */
#ifndef UTILS_HYDRO_H_
#define UTILS_HYDRO_H_

#include <type_traits>

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"
#include "shared/FieldManager.h"

namespace dyablo {
  
/**
 * a dummy swap device routine.
 */
KOKKOS_INLINE_FUNCTION
void swap ( real_t& a, real_t& b ) {
  real_t c = a; a=b; b=c;
}

/**
 * Equation of state:
 * compute pressure p and speed of sound c, from density rho and
 * internal energy eint using the "calorically perfect gas" equation
 * of state : \f$ eint=\frac{p}{\rho (\gamma-1)} \f$
 * Recall that \f$ \gamma \f$ is equal to the ratio of specific heats
 *  \f$ \left[ c_p/c_v \right] \f$.
 * 
 * @param[in]  rho  density
 * @param[in]  eint internal energy
 * @param[out] p    pressure
 * @param[out] c    speed of sound
 */
KOKKOS_INLINE_FUNCTION
void eos(real_t rho,
         real_t eint,
         real_t* p,
         real_t* c,
         const HydroParams& params)
{
  real_t gamma0 = params.settings.gamma0;
  real_t smallp = params.settings.smallp;
  
  *p = FMAX((gamma0 - ONE_F) * rho * eint, rho * smallp);
  *c = SQRT(gamma0 * (*p) / rho);
  
} // eos

/**
 * Convert conservative variables (rho, rho*u, rho*v, e) to 
 * primitive variables (rho,u,v,p)
 * @param[in]  u  conservative variables array
 * @param[out] q  primitive    variables array (allocated in calling routine, size is constant nbvar)
 * @param[out] c  local speed of sound
 */
KOKKOS_INLINE_FUNCTION
void computePrimitives(const HydroState2d &u,
                       real_t             *c,
                       HydroState2d       &q,
                       const HydroParams& params)
{
  real_t gamma0 = params.settings.gamma0;
  real_t smallr = params.settings.smallr;
  real_t smallp = params.settings.smallp;
  
  real_t d, p, ux, uy;
  
  d = fmax(u[ID], smallr);
  ux = u[IU] / d;
  uy = u[IV] / d;
  
  // kinetic energy
  real_t eken = HALF_F * (ux*ux + uy*uy);
  
  // internal energy
  real_t e = u[IP] / d - eken;
  
  // compute pressure and speed of sound
  p = fmax((gamma0 - 1.0) * d * e, d * smallp);
  *c = sqrt(gamma0 * (p) / d);
  
  q[ID] = d;
  q[IP] = p;
  q[IU] = ux;
  q[IV] = uy;
  
} // computePrimitives - 2d

KOKKOS_INLINE_FUNCTION
void computePrimitives(const HydroState3d &u,
                       real_t             *c,
                       HydroState3d       &q,
                       real_t             gamma0,
                       real_t             smallr,
                       real_t             smallp)
{ 
  real_t d, p, ux, uy, uz;
  
  d = fmax(u[ID], smallr);
  ux = u[IU] / d;
  uy = u[IV] / d;
  uz = u[IW] / d;
  
  // kinetic energy
  real_t eken = HALF_F * (ux*ux + uy*uy + uz*uz);
  
  // internal energy
  real_t e = u[IP] / d - eken;
  
  // compute pressure and speed of sound
  p = fmax((gamma0 - 1.0) * d * e, d * smallp);
  *c = sqrt(gamma0 * (p) / d);
  
  q[ID] = d;
  q[IP] = p;
  q[IU] = ux;
  q[IV] = uy;
  q[IW] = uz;  
} 

/**
 * Convert conservative variables (rho, rho*u, rho*v, e) to 
 * primitive variables (rho,u,v,p)
 * @param[in]  u  conservative variables array
 * @param[out] q  primitive    variables array (allocated in calling routine, size is constant nbvar)
 * @param[out] c  local speed of sound
 */
KOKKOS_INLINE_FUNCTION
void computePrimitives(const HydroState3d &u,
                       real_t             *c,
                       HydroState3d       &q,
                       const HydroParams& params)
{
  computePrimitives(u,c,q,params.settings.gamma0,params.settings.smallr,params.settings.smallp);
} // computePrimitives - 3d

/**
 * Compute speed of sound (ideal gas equation of state).
 *
 * @param[in]  u  conservative variables array
 * @param[out] p  pressure
 * @param[out] c speed of sound
 *
 */
KOKKOS_INLINE_FUNCTION
void compute_Pressure_and_SpeedOfSound(const HydroState2d &u,
                                       real_t & pressure,
                                       real_t & c,
                                       const HydroParams& params)
{
  const real_t gamma0 = params.settings.gamma0;
  const real_t smallr = params.settings.smallr;
  const real_t smallp = params.settings.smallp;
  
  real_t d, ux, uy;
  
  d = fmax(u[ID], smallr);
  ux = u[IU] / d;
  uy = u[IV] / d;
  
  // kinetic energy
  const real_t eken = HALF_F * (ux*ux + uy*uy);
  
  // internal
  const real_t eint = u[IP] / d - eken;
  
  // compute pressure
  pressure = fmax((gamma0 - 1.0) * d * eint, d * smallp);
  
  // compute speed of sound
  c = sqrt(gamma0 * (pressure) / d);
  
} // computePressure_and_SpeedOfSound - 2d

/**
 * Compute speed of sound (ideal gas equation of state).
 *
 * @param[in]  u  conservative variables array
 * @param[out] p  pressure
 * @param[out] c speed of sound
 *
 */
KOKKOS_INLINE_FUNCTION
void compute_Pressure_and_SpeedOfSound(const HydroState3d &u,
                                       real_t & pressure,
                                       real_t & c,
                                       const HydroParams& params)
{
  const real_t gamma0 = params.settings.gamma0;
  const real_t smallr = params.settings.smallr;
  const real_t smallp = params.settings.smallp;
  
  real_t d, ux, uy, uz;
  
  d = fmax(u[ID], smallr);
  ux = u[IU] / d;
  uy = u[IV] / d;
  uz = u[IW] / d;
  
  // volumic kinetic energy
  const real_t eken = HALF_F * (ux*ux + uy*uy + uz*uz);
  
  // volumic internal energy
  const real_t eint = u[IP] / d - eken;
  
  // compute pressure
  pressure = fmax((gamma0 - 1.0) * d * eint, d * smallp);
  
  // compute speed of sound
  c = sqrt(gamma0 * (pressure) / d);
  
} // computePressure_and_SpeedOfSound - 3d

} // namespace dyablo

#endif // UTILS_HYDRO_H_
