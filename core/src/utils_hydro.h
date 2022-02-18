/**
 * \file utils_hydro.h
 * \author Pierre Kestener
 */
#ifndef UTILS_HYDRO_H_
#define UTILS_HYDRO_H_

#include <type_traits>

#include "kokkos_shared.h"
#include "HydroState.h"
#include "FieldManager.h"

namespace dyablo {
  
/**
 * a dummy swap device routine.
 */
KOKKOS_INLINE_FUNCTION
void swap ( real_t& a, real_t& b ) {
  real_t c = a; a=b; b=c;
}

KOKKOS_INLINE_FUNCTION
void computePrimitives(const HydroState2d &u,
                       real_t             *c,
                       HydroState2d       &q,
                       real_t             gamma0,
                       real_t             smallr,
                       real_t             smallp)
{ 
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

} // namespace dyablo

#endif // UTILS_HYDRO_H_
