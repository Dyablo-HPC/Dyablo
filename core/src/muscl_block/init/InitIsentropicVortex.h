/**
 * \file InitIsentropicVortex.h
 * \author Maxime Delorme
 **/
#ifndef MUSCL_BLOCK_INIT_ISENTROPIC_VORTEX_H_
#define MUSCL_BLOCK_INIT_ISENTROPIC_VORTEX_H_

#include <limits> // for std::numeric_limits

#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/kokkos_shared.h"
#include "shared/problems/IsentropicVortexParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/amr/AMRmesh.h"

namespace dyablo {
namespace muscl_block {
  
/*************************************************/
/*************************************************/
/*************************************************/

/**
 * Implements the functor to initialize an Isentropic Vortex Advection problem
 *
 * This functor takes as input an already refines mesh and initializes the
 * user data on host. Copying data from host to device should be done outsite.
 
 **/
class InitIsentropicVortexDataFunctor {
private:
  uint32_t nbTeams; //!< number of thread teams
public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams=nbTeams_;};

  // Constructor
  InitIsentropicVortexDataFunctor(std::shared_ptr<AMRmesh> pmesh,
				  HydroParams              params,
				  IsentropicVortexParams   ivParams,
				  id2index_t               fm,
				  blockSize_t              blockSizes,
				  DataArrayBlockHost       Udata_h)
    : pmesh(pmesh), params(params), ivParams(ivParams), fm(fm),
    blockSizes(blockSizes), Udata_h(Udata_h) {};

  // Static function to create and apply the functor to the data
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams              params,
		    ConfigMap                configMap,
		    id2index_t               fm,
		    blockSize_t              blockSizes,
		    DataArrayBlockHost       Udata_h) {
    // Looking up the specific parameters
    IsentropicVortexParams ivParams(configMap);

    // Instantiation of the functor
    InitIsentropicVortexDataFunctor functor(pmesh, params, ivParams, fm, blockSizes, Udata_h);

    // Setting-up execution policy
    uint32_t nbTeams = configMap.getInteger("init", "nbTeams", 16);
    functor.setNbTeams(nbTeams);

    // And initializing on host
    team_policy_t policy(Kokkos::OpenMP(),
			 nbTeams,
			 Kokkos::AUTO());
    Kokkos::parallel_for("dyablo::muscl_block::InitIsentropicVortexDataFunctor",
			 policy, functor);
  }

  // IC operator
  inline
  void operator()(thread_t member) const {
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
      const real_t x0 = pmesh->getCoordinates(iOct)[IX];
      const real_t y0 = pmesh->getCoordinates(iOct)[IY];
      
      const real_t gamma0 = params.settings.gamma0;

      // isentropic vortex parameters
      
      // ambient flow
      const real_t rho_a = ivParams.rho_a;
      // const real_t p_a   = ivParams.p_a;
      const real_t T_a = ivParams.T_a;
      const real_t u_a = ivParams.u_a;
      const real_t v_a = ivParams.v_a;
      // const real_t w_a   = ivParams.w_a;
      
      // vortex center
      real_t vortex_x = ivParams.vortex_x;
      real_t vortex_y = ivParams.vortex_y;

      const bool use_tEnd = this->ivParams.use_tEnd;
      if (use_tEnd) {
	const real_t xmin = this->params.xmin;
	const real_t ymin = this->params.ymin;
	const real_t xmax = this->params.xmax;
	const real_t ymax = this->params.ymax;
	vortex_x += this->ivParams.tEnd * u_a;
	vortex_y += this->ivParams.tEnd * v_a;
	
	// make sure vortex center is inside the box (periodic boundaries for this test)
	vortex_x = fmod(vortex_x, xmax-xmin);
	vortex_y = fmod(vortex_y, ymax-ymin);
      }
    
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        KOKKOS_LAMBDA(const int32_t index) {
	  
          // convert index into ix,iy,iz of local cell inside
          // block
          coord_t iCoord;
          uint32_t& ix = iCoord[IX];
          uint32_t& iy = iCoord[IY];

	  const real_t scale = ivParams.scale;

	  if (params.dimType == TWO_D) {
            iCoord = index_to_coord<2>(index,blockSizes);
          } else {
            iCoord = index_to_coord<3>(index,blockSizes);
          }
	  
	  real_t x = x0 + ix*cellSize + cellSize/2;
          real_t y = y0 + iy*cellSize + cellSize/2;
	  
	  // relative coordinates versus vortex center
	  real_t xp = (x - vortex_x)/scale;
	  real_t yp = (y - vortex_y)/scale;
	  real_t r = sqrt(xp * xp + yp * yp);
	  
	  const real_t beta = ivParams.beta;
	  
	  real_t du = -yp * beta / (2 * M_PI) * exp(0.5 * (1.0 - r * r));
	  real_t dv = xp * beta / (2 * M_PI) * exp(0.5 * (1.0 - r * r));
	  
	  real_t T = T_a - (gamma0 - 1) * beta * beta / (8 * gamma0 * M_PI * M_PI) *
	    exp(1.0 - r * r);
	  real_t rho = rho_a * pow(T / T_a, 1.0 / (gamma0 - 1));

	  Udata_h(index, fm[ID], iOct) = rho;
	  Udata_h(index, fm[IU], iOct) = rho * (u_a + du);
	  Udata_h(index, fm[IV], iOct) = rho * (v_a + dv);
	  Udata_h(index, fm[IP], iOct) = rho * T / (gamma0 - 1.0) +
	    0.5 * rho * (u_a + du) * (u_a + du) +
	    0.5 * rho * (v_a + dv) * (v_a + dv);
	  
	  if (params.dimType == THREE_D)
	    Udata_h(index, fm[IW], iOct) = 0.0;
	}); // end Kokkos::paralle_for
      
      iOct += nbTeams;
    } // end while
  } // end operator()

  //! The mesh on which data will be initialised
  std::shared_ptr<AMRmesh> pmesh;

  //! Global parameters of the simulation
  HydroParams params;

  //! Isentropic Vortex specific parameters
  IsentropicVortexParams ivParams;

  //! Field manager
  id2index_t fm;

  //! Size of the blocks in the current scheme
  blockSize_t blockSizes;

  //! Data array on which the initialization will be made
  DataArrayBlockHost Udata_h;
};
 
/**
 * Implements the initialization functor to solve the isentropic vortex problem.
 *
 * This functor only performs mesh refinement, no user data init.
 *
 * Initial conditions is refined near initial density gradients.
 *
 */
class InitIsentropicVortexRefineFunctor {
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;
  InitIsentropicVortexRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
				    HydroParams              params,
				    IsentropicVortexParams   ivParams,
				    uint8_t                  level_refine)
    : pmesh(pmesh), params(params), ivParams(ivParams), level_refine(level_refine) {};

  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams              params,
		    ConfigMap                configMap,
		    uint8_t                  level_refine) {
    // Fetching problem parameters
    IsentropicVortexParams ivParams(configMap);

    // Refinement
    InitIsentropicVortexRefineFunctor functor(pmesh, params, ivParams, level_refine);

    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());
    Kokkos::parallel_for("dyablo::muscl_block::InitIsentropicVortexRefineFunctor", policy, functor);
  }

  void operator()(const uint32_t &iOct) const {
    uint8_t level = pmesh->getLevel(iOct);

    if (level == level_refine) {
      bool should_refine = false;

      should_refine = true;

      if (should_refine)
	pmesh->setMarker(iOct, 1);
    } // end if level == level_refine
  } // end operator()

  //! Mesh to refine
  std::shared_ptr<AMRmesh> pmesh;

  //! Global simulation parameters
  HydroParams params;

  //! Specific simulation parameters
  IsentropicVortexParams ivParams;

  //! Level to refine
  int level_refine;
};
  
}
}

#endif
