/**
 * \file RTParams.h
 * \author Maxime Delorme
 */
#ifndef RT_PARAMS_H_
#define RT_PARAMS_H_

#include "utils/config/ConfigMap.h"

/**
 * Rayleigh-Taylor instability test parameters.
 */
enum RTInitType {RT_VELOCITY, RT_BOUNDARY};
struct RTParams {
  
  // blast problem parameters
  real_t rt_y;
  real_t amplitude;
  real_t modes_x;
  real_t modes_z;
  real_t rho_up;
  real_t rho_down;
  real_t P0;
  
  RTParams(ConfigMap& configMap)
  {
    
    rt_y      = configMap.getFloat("rayleigh_taylor", "interface_y", 0.5);
    amplitude = configMap.getFloat("rayleigh_taylor", "amplitude", 0.01);
    modes_x   = configMap.getFloat("rayleigh_taylor", "modes_x", 1.0);
    modes_z   = configMap.getFloat("rayleigh_taylor", "modes_z", 1.0);
    rho_up    = configMap.getFloat("rayleigh_taylor", "rho_up", 2.0);
    rho_down  = configMap.getFloat("rayleigh_taylor", "rho_down", 1.0);
    P0        = configMap.getFloat("rayleigh_taylor", "P0", 2.5);
  }

}; // struct BlastParams

#endif // BLAST_PARAMS_H_
