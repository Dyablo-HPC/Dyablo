/**
 * \file FourQuadrantParams.h
 * \author Maxime Delorme
 **/
#ifndef FOURQUADRANT_PARAMS_H_
#define FOURQUADRANT_PARAMS_H_

#include "utils/config/ConfigMap.h"

/**
 * Four-Quadrant problem test parameters
 **/
struct FourQuadrantParams {

  real_t xt, yt;
  int configNumber;

  FourQuadrantParams(ConfigMap &configMap) {
    xt = configMap.getFloat("riemann2d","x",0.8);
    yt = configMap.getFloat("riemann2d","y",0.8);
    configNumber = configMap.getInteger("riemann2d","config_number",0);
  }

}; // struct FourQuadrantParameters

#endif

