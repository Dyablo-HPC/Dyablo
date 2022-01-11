/**
 * \file GreshoVortexParams.h
 * \author Pierre Kestener
 *
 */
#ifndef GRESHO_VORTEX_PARAMS_H_
#define GRESHO_VORTEX_PARAMS_H_

#include "utils/config/ConfigMap.h"

/**
 * The Gresho problem is a rotating vortex problem independent of time
 * for the case of inviscid flow (Euler equations).
 *
 * reference : https://www.cfd-online.com/Wiki/Gresho_vortex
 */
struct GreshoVortexParams {

  real_t rho0;
  real_t Ma;

  // advection velocity (optional)
  real_t u, v, w;
  
  GreshoVortexParams(ConfigMap& configMap)
  {

    rho0  = configMap.getValue<real_t>("Gresho", "rho0", 1.0);
    Ma    = configMap.getValue<real_t>("Gresho", "Ma",   0.1);

    u     = configMap.getValue<real_t>("Gresho", "u",   0.0);
    v     = configMap.getValue<real_t>("Gresho", "v",   0.0);
    w     = configMap.getValue<real_t>("Gresho", "w",   0.0);
    
  } // GreshoVortexParams
  
}; // struct GreshoVortexParams

#endif // GRESHO_VORTEX_PARAMS_H_
