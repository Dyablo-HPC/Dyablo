/**
 * \file IsentropicVortexParams.h
 * \author Pierre Kestener
 *
 */
#ifndef ISENTROPIC_VORTEX_PARAMS_H_
#define ISENTROPIC_VORTEX_PARAMS_H_

#include "utils/config/ConfigMap.h"

/**
 * isentropic vortex advection test parameters.
 */
struct IsentropicVortexParams {

  // isentropic vortex ambient flow
  real_t rho_a;
  real_t p_a;
  real_t T_a;
  real_t u_a;
  real_t v_a;
  real_t w_a;

  // vortex center
  real_t vortex_x;
  real_t vortex_y;
  real_t vortex_z;

  // vortex strength
  real_t beta;

  // scale factor
  real_t scale;

  // number of quadrature points (used to compute initial cell-averaged values)
  int nQuadPts;

  //! useful to compute solution at final time
  bool use_tEnd;
  real_t tEnd;
  
  IsentropicVortexParams(ConfigMap& configMap)
  {

    double xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0);
    double ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0);
    double zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0);

    double xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0);
    double ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0);
    double zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0);
    
    rho_a  = configMap.getValue<real_t>("isentropic_vortex","density_ambient", 1.0);
    p_a  = configMap.getValue<real_t>("isentropic_vortex","pressure_ambient", 1.0);
    T_a  = configMap.getValue<real_t>("isentropic_vortex","temperature_ambient", 1.0);
    u_a  = configMap.getValue<real_t>("isentropic_vortex","vx_ambient", 1.0);
    v_a  = configMap.getValue<real_t>("isentropic_vortex","vy_ambient", 1.0);
    w_a  = configMap.getValue<real_t>("isentropic_vortex","vz_ambient", 1.0);

    vortex_x = configMap.getValue<real_t>("isentropic_vortex","center_x", (xmin+xmax)/2);
    vortex_y = configMap.getValue<real_t>("isentropic_vortex","center_y", (ymin+ymax)/2);
    vortex_z = configMap.getValue<real_t>("isentropic_vortex","center_z", (zmin+zmax)/2);

    beta = configMap.getValue<real_t>("isentropic_vortex","strength",5.0);
    scale = configMap.getValue<real_t>("isentropic_vortex","scale",1.0);

    nQuadPts = configMap.getValue<int>("isentropic_vortex", "num_quadrature_points",4);

    // default value is false, meaning we compute the initial value (t=0)
    use_tEnd = configMap.getValue<bool>("isentropic_vortex", "use_tEnd", false);
    tEnd     = configMap.getValue<real_t>("run", "tEnd", 1.0);
  }

}; // struct IsentropicVortexParams

#endif // ISENTROPIC_VORTEX_PARAMS_H_
