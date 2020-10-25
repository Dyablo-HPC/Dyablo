/**
 * \file InitShuOsher.h
 * \author Maxime Delorme
 **/
#ifndef MUSCL_BLOCK_SHU_OSHER_H_
#define MUSCL_BLOCK_SHU_OSHER_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace dyablo {
namespace muscl_block {

/**
 * Functor to initialize the Shu-Osher problem
 *
 * This functor takes as input an already refined mesh and initializes the user 
 * data on host. Copying the data from host to device should be done outside
 *
 **/
class InitShuOsherDataFunctor {
private:
  uint32_t nbTeams; //!< Number of thread teams;
  
public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<uint32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams=nbTeams_;};

  // Constructor of the functor
  InitShuOsherDataFunctor(std::shared_ptr<AMRmesh> pmesh,
			  HydroParams              params,
			  id2index_t               fm,
			  blockSize_t              blockSizes,
			  DataArrayBlockHost       Udata_h)
    : pmesh(pmesh), params(params), fm(fm), blockSizes(blockSizes), Udata_h(Udata_h) {};

  // Static function that initializes and applies the functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams              params,
		    ConfigMap                configMap,
		    id2index_t               fm,
		    blockSize_t              blockSizes,
		    DataArrayBlockHost       Udata_h) {
    // Instantiation of the functor
    InitShuOsherDataFunctor functor(pmesh, params, fm, blockSizes, Udata_h);

    // And applying it to the mesh
    uint32_t nbTeams = configMap.getInteger("init", "nbTeams", 16);
    functor.setNbTeams(nbTeams);

    team_policy_t policy(Kokkos::OpenMP(),
			 nbTeams,
			 Kokkos::AUTO());

    Kokkos::parallel_for("dyablo::muscl_block::InitShuOsherDataFunctor",
			 policy, functor);
  }

  // Actual initialization operator
  void operator()(thread_t member) const {
    uint32_t iOct = member.league_rank();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    const bool twoD   = (params.dimType==TWO_D);
    const bool threeD = !twoD;

    uint32_t nbCells = twoD ? bx*by : bx*by*bz;


    const real_t gamma0 = params.settings.gamma0;

    // Looping over all octants
    while(iOct <  pmesh->getNumOctants()) {

      // get octant size
      const real_t octSize = pmesh->getSize(iOct);

      const real_t dx = octSize/bx;

      // coordinates of the lower left corner
      const real_t x0 = pmesh->getNode(iOct, 0)[IX];

      // Iterating on all cells
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        [=](const int32_t index) {
	  coord_t iCoord;
          uint32_t& ix = iCoord[IX];                   

          if (twoD) {
            iCoord = index_to_coord<2>(index,blockSizes);
          } else {
            iCoord = index_to_coord<3>(index,blockSizes);
          }

          real_t x = x0 + ix*dx + dx/2;
	  
	  const real_t rho1 = 27.0/7;
	  const real_t rho2 = (1.0+sin(5.0 * M_PI *x)/5);
	  
	  const real_t p1 = 31.0/3;
	  const real_t p2 = 1.0;
	  
	  const real_t v1 = 4*sqrt(35.0)/9;
	  const real_t v2 = 0;
	  
	  real_t d, u, v, w, p;
	  if (x < 0.1) {
	    d = rho1;
	    p = p1/(gamma0-1.0) + 0.5*d*v1*v1;
	    u = d*v1;
	    v = 0.0;
	    w = 0.0;
	  }
	  else {
	    d = rho2;
	    p = p2/(gamma0-1.0) + 0.5*d*v2*v2;
	    u = d*v2;
	    v = 0.0;
	    w = 0.0;
	  }

	  /**
	   * Compute and assign the variables of the problem here !
	   **/
	  Udata_h(index, fm[ID], iOct) = d;
	  Udata_h(index, fm[IU], iOct) = u;
	  Udata_h(index, fm[IV], iOct) = v;
	  Udata_h(index, fm[IP], iOct) = p;
	  if (threeD)
	    Udata_h(index, fm[IW], iOct) = w;
	});
      
      iOct += nbTeams;
    } // end while iOct
  } // end operator()
  	  
  //! Mesh of the simulation
  std::shared_ptr<AMRmesh> pmesh;

  //! Global parameters of the simulation
  HydroParams params;

  //! Field manager
  id2index_t fm;

  //! Block sizes array
  blockSize_t blockSizes;     

  //! Data array on host
  DataArrayBlockHost Udata_h;	  
};

/**
 * Functor to initialize the Shu-Osher problem
 *
 * This functor takes as input an unrefined mesh and performs initial refinement
 **/
class InitShuOsherRefineFunctor {
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;
  InitShuOsherRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
			 HydroParams              params,
			 uint8_t                  level_refine)
    : pmesh(pmesh), params(params), level_refine(level_refine){};

  // Applying the functor to the mesh
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap                configMap,
		    HydroParams              params,
		    uint8_t level_refine) {

    // iterate functor for refinement
    InitShuOsherRefineFunctor functor(pmesh, params, level_refine);

    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());	
    Kokkos::parallel_for("dyablo::muscl_block::InitShuOsherRefineFunctor", policy, functor);
  }

  void operator()(const uint32_t &iOct) const {
    uint8_t level = pmesh->getLevel(iOct);
    
    if (level == level_refine) {
      std::array<double,3> center = pmesh->getCenter(iOct);
      const real_t x = center[0];
      
      double cellSize2 = pmesh->getSize(iOct)*0.75;
      
      bool should_refine = false;

      // Refining around discontinuity
      if (x+cellSize2 >= 0.5 and x-cellSize2 < 0.5)
	should_refine = true;
      
      if (should_refine)
        pmesh->setMarker(iOct, 1);
    } // end if level == level_refine
  } // end operator()

  //! The mesh to refine
  std::shared_ptr<AMRmesh> pmesh;

  //! Global simulation parameters
  HydroParams params;

  //! Level to refine
  uint8_t level_refine;
};

} // end namespace muscl_block
} // end namespace dyablo


#endif
