/**
 * \file InitKelvinHelmholtz.h
 * \author Maxime Delorme
 **/
#ifndef MUSCL_BLOCK_KELVIN_HELMHOLTZ_H_
#define MUSCL_BLOCK_KELVIN_HELMHOLTZ_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/KHParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// kokkos random numbers generator
#include <Kokkos_Random.hpp>

namespace dyablo {
namespace muscl_block {

/**
 * Implements the functor to initialize a Kelvin Helmholtz instability
 *
 * This functor takes as input an already refines mesh and initializes the
 * user data on host. Copying data from host to device should be done outsite.
 
 **/
class InitKelvinHelmholtzDataFunctor {
private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;
  using rand_type = Kokkos::Random_XorShift64_Pool<Device>::generator_type;
  
  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;};

  // Constructor of the functor
  InitKelvinHelmholtzDataFunctor(std::shared_ptr<AMRmesh> pmesh,
				 HydroParams              params,
				 KHParams                 khParams,
				 id2index_t               fm,
				 blockSize_t              blockSizes,
				 DataArrayBlockHost       Udata_h)
    : pmesh(pmesh), params(params), khParams(khParams),
    fm(fm), blockSizes(blockSizes), Udata_h(Udata_h), rand_pool(khParams.seed) {};

  // Static function that initializes the functor and applies it on the mesh
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams             params,
		    ConfigMap               configMap,
		    id2index_t              fm,
		    blockSize_t             blockSizes,
		    DataArrayBlockHost      Udata_h) {
    
    // Looking up the specific parameters
    KHParams khParams(configMap);

    // Instantiation of the functor
    InitKelvinHelmholtzDataFunctor functor(pmesh, params, khParams, fm, blockSizes, Udata_h);

    // Setting-up the execution policy
    uint32_t nbTeams = configMap.getInteger("init","nbTeams",16);
    functor.setNbTeams(nbTeams);

    // And initializing on host
    team_policy_t policy(Kokkos::OpenMP(),
			 nbTeams,
			 Kokkos::AUTO());

    Kokkos::parallel_for("dyablo::muscl_block::InitKelvinHelmholtzDataFunctor",
			 policy, functor);
  } // end constructor

  // The actual initialization operator
  void operator()(thread_t member) const {
    // Kelvin Helmholtz problem parameters
    const real_t d_in  = khParams.d_in;
    const real_t d_out = khParams.d_out;
    const real_t vflow_in  = khParams.vflow_in;
    const real_t vflow_out = khParams.vflow_out;
    const real_t ampl      = khParams.amplitude;
    const real_t pressure  = khParams.pressure;

    const real_t gamma0 = params.settings.gamma0;
    
    uint32_t iOct = member.league_rank();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    while(iOct <  pmesh->getNumOctants()) {
      
      // get octant size
      const real_t octSize = pmesh->getSize(iOct);

      // the following assumes bx=by=bz
      // TODO : allow non-cubic blocks
      const real_t cellSize = octSize/bx;

      // coordinates of the lower left corner
      const real_t x0 = pmesh->getNode(iOct, 0)[IX];
      const real_t y0 = pmesh->getNode(iOct, 0)[IY];
      const real_t z0 = pmesh->getNode(iOct, 0)[IZ];

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        KOKKOS_LAMBDA(const int32_t index) {
	  
          // convert index into ix,iy,iz of local cell inside
          // block
          coord_t iCoord;
          uint32_t& ix = iCoord[IX];
          uint32_t& iy = iCoord[IY];
          uint32_t& iz = iCoord[IZ];                    

          if (params.dimType == TWO_D) {
            iCoord = index_to_coord<2>(index,blockSizes);
          } else {
            iCoord = index_to_coord<3>(index,blockSizes);
          }

          // compute x,y,z for that cell (cell center)
          real_t x = x0 + ix*cellSize + cellSize/2;
          real_t y = y0 + iy*cellSize + cellSize/2;
          real_t z = z0 + iz*cellSize + cellSize/2;

	  if (khParams.p_rand) {
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
	    
	    Udata_h(index, fm[ID], iOct) = d;
	    Udata_h(index, fm[IU], iOct) = d * u;
	    Udata_h(index, fm[IV], iOct) = d * v;
	    if (params.dimType == THREE_D)
	      Udata_h(index, fm[IW], iOct) = d * w;
	    
	    Udata_h(index, fm[IE], iOct) = pressure/(gamma0-1.0) + 0.5*d*(u*u+v*v+w*w);
	    
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
	    
	    Udata_h(index, fm[ID], iOct) = d;
	    Udata_h(index, fm[IU], iOct) = d * u;
	    Udata_h(index, fm[IV], iOct) = d * v;
	    if (params.dimType == THREE_D)
	      Udata_h(index,fm[IW], iOct) = d * w;
	    Udata_h(index, fm[IP], iOct) = pressure / (gamma0-1.0) + 0.5*d*(u*u+v*v+w*w);
	    
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
	    
	    Udata_h(index, fm[ID], iOct) = d;
	    Udata_h(index, fm[IU], iOct) = d * u;
	    Udata_h(index, fm[IV], iOct) = d * v;
	    if (params.dimType == THREE_D)
	      Udata_h(index, fm[IW], iOct) = d * w;
	    Udata_h(index, fm[IP], iOct) = pressure / (gamma0-1.0) + 0.5*d*(u*u+v*v+w*w);
	  } // khParams.p_rand / khParams.p_sine_rob / khParams.p_sine
        }); // end TeamVectorRange

      iOct += nbTeams;

    } // end while
  } // end operator()

  //! Mesh of the simulation
  std::shared_ptr<AMRmesh> pmesh;

  //! Global parameters of the simulation
  HydroParams params;

  //! Parameters of the KH instability ICs
  KHParams khParams;

  //! Field manager
  id2index_t fm;

  //! Block sizes array
  blockSize_t blockSizes;     

  //! Data array on host
  DataArrayBlockHost Udata_h;

  //! Random number generator
  Kokkos::Random_XorShift64_Pool<Device> rand_pool;;
};
 
/**
 * This functor performs mesh refinement (without init) for the
 * Kelvin Helmholtz instability problem
 *
 * Initial conditions are refined near initial density gradients.
 */
class InitKelvinHelmholtzRefineFunctor {
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;
  InitKelvinHelmholtzRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
				   HydroParams              params,
				   KHParams                 khParams,
				   uint8_t                  level_refine)
    : pmesh(pmesh), params(params), khParams(khParams), level_refine(level_refine){};

  // Applying the functor to the mesh
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap                configMap,
		    HydroParams              params,
		    uint8_t level_refine) {
    KHParams khParams(configMap);

    // iterate functor for refinement
    InitKelvinHelmholtzRefineFunctor functor(pmesh, params, khParams, level_refine);

    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());	
    Kokkos::parallel_for("dyablo::muscl_block::InitKelvinHelmholtzRefineFunctor", policy, functor);
  }

  void operator()(const uint32_t &iOct) const {
    
    uint8_t level = pmesh->getLevel(iOct);

    if (level == level_refine) {
      std::array<double,3> center = pmesh->getCenter(iOct);
      const real_t y = center[1];
      const real_t z = center[2];

      double cellSize2 = pmesh->getSize(iOct)*0.75;
      
      bool should_refine = false;

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
        pmesh->setMarker(iOct, 1);
    } // end if level == level_refine
  } // end operator()

  //! The mesh to refine
  std::shared_ptr<AMRmesh> pmesh;

  //! Global simulation parameters
  HydroParams params;

  //! Kelvin-Helmholtz instability parameters
  KHParams khParams;

  //! Level to refine
  uint8_t level_refine;
};

} // namespace dyablo
} // namespace muscl_block

#endif
