/**
 * \file ImplodeParams.h
 * \author Pierre Kestener
 */
#ifndef IMPLODE_PARAMS_H_
#define IMPLODE_PARAMS_H_

#include "utils/config/ConfigMap.h"

/**
 * Implode test parameters.
 */
struct ImplodeParams {

  // outer parameters
  real_t rho_out;
  real_t p_out;
  real_t u_out;
  real_t v_out;
  real_t w_out;
  real_t Bx_out;
  real_t By_out;
  real_t Bz_out;

  // inner parameters
  real_t rho_in;
  real_t p_in;
  real_t u_in;
  real_t v_in;
  real_t w_in;
  real_t Bx_in;
  real_t By_in;
  real_t Bz_in;

  // shape of regions: 0 - diagonal; 1 - curved
  int shape;

  // if true slightly change init condition to be non trivial
  // gradiant along domain diagonal
  bool debug;

  ImplodeParams(ConfigMap& configMap)
  {

    rho_out  = configMap.getValue<real_t>("implode","density_outer", 1.0);
    p_out  = configMap.getValue<real_t>("implode","pressure_outer", 1.0);
    u_out  = configMap.getValue<real_t>("implode","vx_outer", 0.0);
    v_out  = configMap.getValue<real_t>("implode","vy_outer", 0.0);
    w_out  = configMap.getValue<real_t>("implode","vz_outer", 0.0);
    Bx_out  = configMap.getValue<real_t>("implode","Bx_outer", 0.0);
    By_out  = configMap.getValue<real_t>("implode","By_outer", 0.0);
    Bz_out  = configMap.getValue<real_t>("implode","Bz_outer", 0.0);

    rho_in  = configMap.getValue<real_t>("implode","density_inner", 0.125);
    p_in  = configMap.getValue<real_t>("implode","pressure_inner", 0.14);
    u_in  = configMap.getValue<real_t>("implode","vx_inner", 0.0);
    v_in  = configMap.getValue<real_t>("implode","vy_inner", 0.0);
    w_in  = configMap.getValue<real_t>("implode","vz_inner", 0.0);
    Bx_in  = configMap.getValue<real_t>("implode","Bx_inner", 0.0);
    By_in  = configMap.getValue<real_t>("implode","By_inner", 0.0);
    Bz_in  = configMap.getValue<real_t>("implode","Bz_inner", 0.0);

    shape = configMap.getValue<int>("implode", "shape_region",0);
    
    debug = configMap.getValue<bool>("implode", "debug", false);
  }

}; // struct ImplodeParams

#endif // IMPLODE_PARAMS_H_
