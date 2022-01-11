/**
 * \file BlastParams.h
 * \author Pierre Kestener
 */
#ifndef BLAST_PARAMS_H_
#define BLAST_PARAMS_H_

#include "utils/config/ConfigMap.h"

/**
 * Blast test parameters.
 */
struct BlastParams {

  // blast problem parameters
  real_t blast_radius;
  real_t blast_center_x;
  real_t blast_center_y;
  real_t blast_center_z;
  real_t blast_density_in;
  real_t blast_density_out;
  real_t blast_pressure_in;
  real_t blast_pressure_out;
  int    blast_nx;
  int    blast_ny;
  int    blast_nz;

  BlastParams(ConfigMap& configMap)
  {

    double xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0);
    double ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0);
    double zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0);

    double xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0);
    double ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0);
    double zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0);
    
    blast_radius   = configMap.getValue<real_t>("blast","radius", (xmin+xmax)/2.0/10);
    blast_center_x = configMap.getValue<real_t>("blast","center_x", (xmin+xmax)/2);
    blast_center_y = configMap.getValue<real_t>("blast","center_y", (ymin+ymax)/2);
    blast_center_z = configMap.getValue<real_t>("blast","center_z", (zmin+zmax)/2);
    blast_density_in  = configMap.getValue<real_t>("blast","density_in", 1.0);
    blast_density_out = configMap.getValue<real_t>("blast","density_out", 1.2);
    blast_pressure_in  = configMap.getValue<real_t>("blast","pressure_in", 10.0);
    blast_pressure_out = configMap.getValue<real_t>("blast","pressure_out", 0.1);

    blast_nx = configMap.getValue<int>("blast", "blast_nx", 1);
    blast_ny = configMap.getValue<int>("blast", "blast_ny", 1);
    blast_nz = configMap.getValue<int>("blast", "blast_nz", 1);
  }

}; // struct BlastParams

#endif // BLAST_PARAMS_H_
