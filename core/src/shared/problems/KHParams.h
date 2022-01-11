/**
 * \file KHParams.h
 * \author Pierre Kestener
 */
#ifndef KELVIN_HELMHOLTZ_PARAMS_H_
#define KELVIN_HELMHOLTZ_PARAMS_H_

#include "utils/config/ConfigMap.h"
#include "utils/mpi/GlobalMpiSession.h"

//#include <cstdlib> // for srand

/**
 * A small structure to hold parameters passed to a Kokkos functor,
 * for initializing the Kelvin-Helmholtz instability init condition.
 *
 * p_sine, p_sine_robertson and p_rand specifiy which type of perturbation is
 * used to seed the instability.
 */
struct KHParams {

  // Kelvin-Helmholtz problem parameters
  real_t d_in;  //! density in
  real_t d_out; //! density out
  real_t pressure;
  bool p_sine; //! sinus perturbation
  bool p_sine_rob; //! sinus perturbation "a la Robertson"
  bool p_rand; //! random perturbation

  real_t vflow_in;
  real_t vflow_out;

  int seed;
  real_t amplitude; //! perturbation amplitude
  real_t outer_size;
  real_t inner_size;

  // for sine perturbation "a la Robertson"
  int    mode;
  real_t w0;
  real_t delta;
  
  KHParams(ConfigMap& configMap)
  {

    d_in  = configMap.getValue<real_t>("KH", "d_in", 1.0);
    d_out = configMap.getValue<real_t>("KH", "d_out", 2.0);

    pressure = configMap.getValue<real_t>("KH", "pressure", 10.0);

    p_sine     = configMap.getValue<bool>("KH", "perturbation_sine", false);
    p_sine_rob = configMap.getValue<bool>("KH", "perturbation_sine_robertson", true);
    p_rand     = configMap.getValue<bool>("KH", "perturbation_rand", false);

    vflow_in  = configMap.getValue<real_t>("KH", "vflow_in",  -0.5);
    vflow_out = configMap.getValue<real_t>("KH", "vflow_out",  0.5);

    if (p_rand) {
      // choose a different random seed per mpi rank
      seed = configMap.getValue<int>("KH", "rand_seed", 12);
      int mpiRank = dyablo::GlobalMpiSession::get_comm_world().MPI_Comm_rank();
      seed *= (mpiRank+1);      
    }

    amplitude = configMap.getValue<real_t>("KH", "amplitude", 0.1);


    if (p_sine_rob or p_sine) {
      
      // perturbation mode number
      inner_size = configMap.getValue<real_t>("KH","inner_size", 0.2);

      mode       = configMap.getValue<int>("KH", "mode", 2);
      w0         = configMap.getValue<real_t>("KH", "w0", 0.1);
      delta      = configMap.getValue<real_t>("KH", "delta", 0.03);
    }
    
    
  } // KHParams

}; // struct KHParams

#endif // KELVIN_HELMHOLTZ_PARAMS_H_
