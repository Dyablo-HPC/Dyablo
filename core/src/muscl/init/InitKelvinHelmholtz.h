/**
 * \file InitKelvinHelmholtz.h
 * \author Pierre Kestener
 */
#ifndef HYDRO_INIT_KELVIN_HELMHOLTZ_H_
#define HYDRO_INIT_KELVIN_HELMHOLTZ_H_

#include <limits> // for std::numeric_limits

#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/kokkos_shared.h"
#include "shared/problems/KHParams.h"

#include "bitpit_PABLO.hpp"
#include "shared/amr/AMRmesh.h"

// kokkos random numbers generator
#include <Kokkos_Random.hpp>

namespace dyablo {
namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve blast problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 *
 *
 * Initial conditions is refined near strong density gradients.
 *
 */
class InitKelvinHelmholtzDataFunctor {

public:
  InitKelvinHelmholtzDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                                 HydroParams params, 
                                 KHParams khParams,
                                 id2index_t fm, 
                                 DataArray Udata)
      : pmesh(pmesh), 
        params(params),
        khParams(khParams),
        fm(fm),
        Udata(Udata),
        rand_pool(khParams.seed)        
        {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh, 
                    HydroParams params,
                    ConfigMap configMap,
                    id2index_t fm,
                    DataArray Udata) 
  {
    
    KHParams khParams = KHParams(configMap);
    
    // data init functor
    InitKelvinHelmholtzDataFunctor functor(pmesh, params, khParams, fm,
                                           Udata);

    Kokkos::parallel_for("dyablo::muscl::InitKelvinHelmholtzDataFunctor",
                         Kokkos::RangePolicy<Kokkos::OpenMP>(0, pmesh->getNumOctants()), functor);
  }

  void operator()(const size_t &i) const {

    // Kelvin Helmholtz problem parameters
    const real_t d_in  = khParams.d_in;
    const real_t d_out = khParams.d_out;
    const real_t vflow_in  = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl      = khParams.amplitude;
    const real_t pressure  = khParams.pressure;

    const real_t gamma0 = params.settings.gamma0;

    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double, 3> center = pmesh->getCenter(i);

    const real_t x = center[0];
    const real_t y = center[1];
    const real_t z = center[2];


    if (khParams.p_rand) {
      
      // get random number state
      rand_type rand_gen = rand_pool.get_state();

      real_t d, u, v, w;

      real_t tmp = params.dimType == TWO_D ? y : z;

      if ( tmp < 0.25 or tmp > 0.75 ) {
	
	d = d_out;
	u = vflow_out;
	v = 0.0;
        w = 0.0;

      } else {
	
	d = d_in;
	u = vflow_in;
	v = 0.0;
        w = 0.0;
	
      }

      u += ampl * (rand_gen.drand() - 0.5);
      v += ampl * (rand_gen.drand() - 0.5);

      if (params.dimType==THREE_D)
        w += ampl * (rand_gen.drand() - 0.5);
        
      Udata(i,fm[ID]) = d;
      Udata(i,fm[IU]) = d * u;
      Udata(i,fm[IV]) = d * v;
      if (params.dimType == THREE_D)
        Udata(i,fm[IW]) = d * w;
        
      Udata(i,fm[IE]) = pressure/(gamma0-1.0) + 0.5*d*(u*u+v*v+w*w);

      // free random number
      rand_pool.free_state(rand_gen);
      
    } else if (khParams.p_sine_rob) {

      const int    n     = khParams.mode;
      const real_t w0    = khParams.w0;
      const real_t delta = khParams.delta;

      const double tmp1 = 0.25;
      const double tmp2 = 0.75;

      const double rho1 = d_in;
      const double rho2 = d_out;

      const double v1 = vflow_in;
      const double v2 = vflow_out;

      const double v1y = vflow_in/2;
      const double v2y = vflow_out/2;

      const double tmp = params.dimType == TWO_D ? y : z;

      const double ramp = 
	1.0 / ( 1.0 + exp( 2*(tmp-tmp1)/delta ) ) +
	1.0 / ( 1.0 + exp( 2*(tmp2-tmp)/delta ) );

      const real_t d = rho1 + ramp*(rho2-rho1);
      const real_t u = v1   + ramp*(v2-v1);
      const real_t v = params.dimType == TWO_D ? 
        w0 * sin(n*M_PI*x) :
        v1y   + ramp*(v2y-v1y);
      const real_t w = params.dimType == TWO_D ?
        0.0 :
        w0 * sin(n*M_PI*x) * sin(n*M_PI*y);

      Udata(i,fm[ID]) = d;
      Udata(i,fm[IU]) = d * u;
      Udata(i,fm[IV]) = d * v;
      if (params.dimType == THREE_D)
        Udata(i,fm[IW]) = d * w;
      Udata(i,fm[IP]) = pressure / (gamma0-1.0) + 0.5*d*(u*u+v*v+w*w);
      
    } else if (khParams.p_sine) {
      const int    n     = khParams.mode;
      const real_t w0    = khParams.w0;

      const double rho1 = d_in;
      const double rho2 = d_out;
      
      const double tmp = params.dimType == TWO_D ? y : z;

      const double v1 = vflow_in;
      const double v2 = vflow_out;

      real_t d, u, v;
      if (tmp < 0.25 or tmp > 0.75) {
	d = rho2;
	u = v2;
      }
      else {
	d = rho1;
	u = v1;
      }
      
      v = w0 * sin(n*2.0*M_PI*x);

      const real_t w = 0.0;

      Udata(i,fm[ID]) = d;
      Udata(i,fm[IU]) = d * u;
      Udata(i,fm[IV]) = d * v;
      if (params.dimType == THREE_D)
        Udata(i,fm[IW]) = d * w;
      Udata(i,fm[IP]) = pressure / (gamma0-1.0) + 0.5*d*(u*u+v*v+w*w);
    }
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams params;
  KHParams khParams;
  id2index_t fm;
  DataArray Udata;

  // random number generator
  Kokkos::Random_XorShift64_Pool<Device> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<Device>::generator_type rand_type;

}; // InitKelvinHelmholtzDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve blast problem.
 *
 * This functor only performs mesh refinement, no user data init.
 *
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitKelvinHelmholtzDataFunctor
 *
 */
class InitKelvinHelmholtzRefineFunctor {

public:
  InitKelvinHelmholtzRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                                   HydroParams params, KHParams khParams,
                                   int level_refine)
      : pmesh(pmesh), params(params), khParams(khParams),
        level_refine(level_refine){};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh, 
                    ConfigMap configMap,
                    HydroParams params, 
                    int level_refine) 
  {

    KHParams blastParams = KHParams(configMap);

    // iterate functor for refinement
    InitKelvinHelmholtzRefineFunctor functor(pmesh, params, blastParams,
                                             level_refine);
    Kokkos::parallel_for("dyablo::muscl::InitKelvinHelmholtzRefineFunctor", 
                         Kokkos::RangePolicy<Kokkos::OpenMP>(0, pmesh->getNumOctants()), functor);
  }

  void operator()(const size_t &i) const 
  {

    // constexpr double eps = 0.005;

    // get cell level
    uint8_t level = pmesh->getLevel(i);

    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double, 3> center = pmesh->getCenter(i);

      //const real_t x = center[0];
      const real_t y = center[1];
      const real_t z = center[2];

      double cellSize2 = pmesh->getSize(i) * 0.75;

      bool should_refine = false;

      // refine near discontinuities

      if (params.dimType == TWO_D) {

        if ( (y + cellSize2 >= 0.25 and y - cellSize2 < 0.25) or
             (y + cellSize2 >= 0.75 and y - cellSize2 < 0.75) )
          should_refine = true;
      
      } else { // THREE_D
      
        if ( (z + cellSize2 >= 0.25 and z - cellSize2 < 0.25) or
             (z + cellSize2 >= 0.75 and z - cellSize2 < 0.75) )
          should_refine = true;

      }

      if (should_refine)
        pmesh->setMarker(i, 1);

    } // end if level == level_refine

  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams params;
  KHParams khParams;
  int level_refine;

}; // InitKelvinHelmholtzRefineFunctor

} // namespace muscl

} // namespace dyablo

#endif // HYDRO_INIT_KELVIN_HELMHOLTZ_H_
