/**
 * \file InitDoubleMachReflection.h
 * \author Maxime Delorme
 **/

#ifndef MUSCL_BLOCK_DMR_H_
#define MUSCL_BLOCK_DMR_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

// Specific param structure of the problem
#include "shared/problems/DMRParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/amr/AMRmesh.h"

namespace dyablo {
namespace muscl_block {

/**
 * Functor to initialize the Double Mach Reflection problem
 *
 * This functor takes as input an already refined mesh and initializes the user 
 * data on host. Copying the data from host to device should be done outside
 **/
class InitDoubleMachReflectionDataFunctor {
private:
  uint32_t nbTeams; //!< Number of thread teams;
  
public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<uint32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams=nbTeams_;};

  // Constructor of the functor
  InitDoubleMachReflectionDataFunctor(std::shared_ptr<AMRmesh> pmesh,
		       HydroParams              params,
		       DMRParams                dmrParams,		       
		       id2index_t               fm,
		       blockSize_t              blockSizes,
		       DataArrayBlockHost       Udata_h)
    : pmesh(pmesh), params(params), dmrParams(dmrParams),
    fm(fm), blockSizes(blockSizes), Udata_h(Udata_h) {};

  // Static function that initializes and applies the functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams              params,
		    ConfigMap                configMap,
		    id2index_t               fm,
		    blockSize_t              blockSizes,
		    DataArrayBlockHost       Udata_h) {
    // Loading up specific parameters if necessary
    DMRParams dmrParams(configMap);

    // Instantiation of the functor
    InitDoubleMachReflectionDataFunctor functor(pmesh, params, dmrParams, fm, blockSizes, Udata_h);

    // And applying it to the mesh
    uint32_t nbTeams = configMap.getValue<uint32_t>("init", "nbTeams", 16);
    functor.setNbTeams(nbTeams);

    team_policy_t policy(Kokkos::OpenMP(),
			 nbTeams,
			 Kokkos::AUTO());

    Kokkos::parallel_for("dyablo::muscl_block::InitDoubleMachReflectionDataFunctor",
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

    /**
     * Non specific init, if you need to precalculate things
     * that do not depend on the position in the mesh, do it here !
     **/
    const real_t xs    = dmrParams.xs;
    const real_t tant  = tanf(dmrParams.angle);

    // Looping over all octants
    while(iOct <  pmesh->getNumOctants()) {

      // get octant size
      const real_t octSize = pmesh->getSize(iOct);

      const real_t dx = octSize/bx;
      const real_t dy = octSize/by;
      
      // coordinates of the lower left corner
      const real_t x0 = pmesh->getCoordinates(iOct)[IX];
      const real_t y0 = pmesh->getCoordinates(iOct)[IY];

      // Iterating on all cells
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        [&](const int32_t index) {
	  coord_t iCoord;
          uint32_t& ix = iCoord[IX];
          uint32_t& iy = iCoord[IY];                 

          if (twoD) {
            iCoord = index_to_coord<2>(index,blockSizes);
          } else {
            iCoord = index_to_coord<3>(index,blockSizes);
          }

          real_t x = x0 + ix*dx + dx/2;
          real_t y = y0 + iy*dy + dy/2;

	  real_t d, u, v, w, p;
	  if (y < tant * (x - xs)) {
	    d = dmrParams.dL;
	    p = dmrParams.pL;
	    u = dmrParams.uL;
	    v = dmrParams.vL;
	    w = dmrParams.wL;
	  }
	  else {
	    d = dmrParams.dR;
	    p = dmrParams.pR;
	    u = dmrParams.uR;
	    v = dmrParams.vR;
	    w = dmrParams.wR;
	  }

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

  //! Specific parameters
  DMRParams dmrParams;

  //! Field manager
  id2index_t fm;

  //! Block sizes array
  blockSize_t blockSizes;     

  //! Data array on host
  DataArrayBlockHost Udata_h;	  
};

/**
 * Functor to initialize the DoubleMachReflection problem
 *
 * This functor takes as input an unrefined mesh and performs initial refinement
 *
 * The parameter structures DMRParams (if necessary) should be located in 
 * src/shared/problems
 **/
class InitDoubleMachReflectionRefineFunctor {
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;
  InitDoubleMachReflectionRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
			 HydroParams              params,
			 DMRParams              dmrParams,
			 uint8_t                  level_refine)
    : pmesh(pmesh), params(params), dmrParams(dmrParams), level_refine(level_refine){};

  // Applying the functor to the mesh
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap                configMap,
		    HydroParams              params,
		    uint8_t level_refine) {
    DMRParams dmrParams(configMap);

    // iterate functor for refinement
    InitDoubleMachReflectionRefineFunctor functor(pmesh, params, dmrParams, level_refine);

    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());	
    Kokkos::parallel_for("dyablo::muscl_block::InitDoubleMachReflectionRefineFunctor", policy, functor);
  }

  void operator()(const uint32_t &iOct) const {
    uint8_t level = pmesh->getLevel(iOct);
    
    /**
     * The functor is called once per level to refine (from level_min to level_max-1)
     * The decision if the cell should be refined should be only made in this test
     * which checks that the current cell is effectively on the current level of 
     * consideration
     **/
    if (level == level_refine) {
      std::array<double,3> center = pmesh->getCenter(iOct);
      const real_t x = center[0];
      const real_t y = center[1];
      
      const real_t xs    = dmrParams.xs;
      const real_t tant  = tanf(dmrParams.angle);
      
      double ref_dx = pmesh->getSize(iOct) * 1.5;
      
      bool should_refine = (fabs(tant * (x-xs) - y) < ref_dx);

      if (should_refine)
        pmesh->setMarker(iOct, 1);
    } // end if level == level_refine
  } // end operator()

  //! The mesh to refine
  std::shared_ptr<AMRmesh> pmesh;

  //! Global simulation parameters
  HydroParams params;

  //! Specific parameters
  DMRParams dmrParams;

  //! Level to refine
  uint8_t level_refine;
};

} // end namespace muscl_block
} // end namespace dyablo

#endif
