/**
 * \file DMRParams.h
 * \author Maxime Delorme
 **/
#ifndef DOUBLE_MACH_REFLECTION_PARAMS_H_
#define DOUBLE_MACH_REFLECTION_PARAMS_H_

/**
 * Structure to hold parameters of the Double Mach Reflection
 **/

struct DMRParams {
  real_t xs;
  real_t angle;
  
  real_t dL;
  real_t pL;
  real_t uL;
  real_t vL;
  real_t wL;

  real_t dR;
  real_t pR;
  real_t uR;
  real_t vR;
  real_t wR;

  DMRParams(ConfigMap& configMap) {
    xs    = configMap.getFloat("DMR", "xs",    1.0/24.0);
    angle = configMap.getFloat("DMR", "angle", 1.0471975511965976); // 60 degrees

    // Pre-shock state
    dL = configMap.getFloat("DMR", "dL", 1.4);
    pL = configMap.getFloat("DMR", "pL", 1.0);
    uL = configMap.getFloat("DMR", "uL", 0.0);
    vL = configMap.getFloat("DMR", "vL", 0.0);
    wL = configMap.getFloat("DMR", "wL", 0.0);

    // Post-shock state
    dR = configMap.getFloat("DMR", "dR", 8.0);
    pR = configMap.getFloat("DMR", "pR", 116.15);
    uR = configMap.getFloat("DMR", "uR", 4.125*sqrt(3));
    vR = configMap.getFloat("DMR", "vR", -4.125);
    wR = configMap.getFloat("DMR", "wR", 0.0);
  }
};

#endif
