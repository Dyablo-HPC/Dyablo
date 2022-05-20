#pragma once

#include "kokkos_shared.h"
#include "HydroState.h"

namespace dyablo {

// NOTE : This file is almost empty, and maybe should be merged with hydrostate ?
  
/**
 * a dummy swap device routine.
 */
KOKKOS_INLINE_FUNCTION
void swap ( real_t& a, real_t& b ) {
  real_t c = a; a=b; b=c;
}

template<typename PrimState, typename ConsState>
KOKKOS_INLINE_FUNCTION
void computePrimitives(const ConsState &u,
                       real_t          *c,
                       PrimState       &q,
                       real_t           gamma0,
                       real_t           smallr,
                       real_t           smallp)
{ 
  real_t d, p, ux, uy, uz;
  
  d = fmax(u.rho, smallr);
  ux = u.rho_u / d;
  uy = u.rho_v / d;
  uz = u.rho_w / d;
  
  // kinetic energy
  real_t eken = HALF_F * (ux*ux + uy*uy + uz*uz);
  
  // internal energy
  real_t e = u.e_tot / d - eken;
  
  // compute pressure and speed of sound
  p = fmax((gamma0 - 1.0) * d * e, d * smallp);
  *c = sqrt(gamma0 * (p) / d);
  
  q.rho = d;
  q.p   = p;
  q.u   = ux;
  q.v   = uy;
  q.w   = uz;  
}

} // namespace dyablo
