/**
 * \file InitTemplate.h
 * \author Maxime Delorme
 *
 * \brif This is a template for initialization of a problem in dyablo.
 *
 * The initial conditions of a problem require two functors:
 * - The first is the Data functor (InitXXXXXDataFunctor) which takes as input
 *   an already refined mesh and fills the Data array accordingly. This is done
 *   on host which means the copy to the device should be done outside of the 
 *   functor.
 * - The second functor is the Refine functor (InitXXXXXRefineFunctor) which
 *   takes an unrefined mesh and performs initial mesh refinement.
 *
 * To make your own problem init, here's a checklist of things to do: 
 *
 * (0-) Make a copy of this file with a clear name of your test (eg InitBlast.h)
 * (1-) Make a problem parameter structure if your problem requires initialization parameters
 *      this structure should go in src/shared/problems/XXXXXXParams.h. 
 *      If you don't need parameters for your ICs, look for xxParams and remove all occurences
 *  2-  Replace all the XXXXX by the name of your problem
 *  3-  Fill in the operator() functions of both InitXXXXXDataFunctor and InitXXXXXRefineFunctor
 * (4-) If not already done, copy and adapt the InitTemplate.cpp file as well !  
 *  5-  Update the HydroInitFunctors.h to add the function defined in InitTemplate.cpp
 *  6-  Update the init method in SolverHydroMusclBlock.cpp to include your init function
 *  7-  Update the CMakeLists.txt in src/muscl_block to account for the new .cpp file
 */

#ifndef MUSCL_BLOCK_XXXXX_H_
#define MUSCL_BLOCK_XXXXX_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

// Specific param structure of the problem
#include "shared/problems/XXXXXParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// Additional includes here


namespace dyablo {
namespace muscl_block {

/**
 * Functor to initialize the XXXXX problem
 *
 * This functor takes as input an already refined mesh and initializes the user 
 * data on host. Copying the data from host to device should be done outside
 *
 * The parameter structures XXXXXParams (if necessary) should be located in 
 * src/shared/problems
 **/
class InitXXXXXDataFunctor {
private:
  uint32_t nbTeams; //!< Number of thread teams;
  
public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<uint32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams=nbTeams_;};

  // Constructor of the functor
  InitXXXXXDataFunctor(std::shared_ptr<AMRmesh> pmesh,
		       HydroParams              params,
		       XXXXXParams              xxparams,		       
		       id2index_t               fm,
		       blockSize_t              blockSizes,
		       DataArrayBlockHost       Udata_h)
    : pmesh(pmesh), params(params), xxparams(xxparams),
    fm(fm), blockSizes(blockSizes), Udata_h(Udata_h) {};

  // Static function that initializes and applies the functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams              params,
		    ConfigMap                configMap,
		    id2index_t               fm,
		    blockSize_t              blockSizes,
		    DataArrayBlockHost       Udata_h) {
    // Loading up specific parameters if necessary
    XXXXXParams xxParams(configMap);

    // Instantiation of the functor
    InitXXXXXDataFunctor functor(pmesh, params, xxParams, fm, blockSizes, Udata_h);

    // And applying it to the mesh
    uint32_t nbTeams = configMap.getInteger("init", "nbTeams", 16);
    functor.setNbTeams(nbTeams);

    team_policy_t policy(Kokkos::OpenMP(),
			 nbTeams,
			 Kokkos::AUTO());

    Kokkos::parallel_for("dyablo::muscl_block::InitXXXXXDataFunctor",
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


    // Looping over all octants
    while(iOct <  pmesh->getNumOctants()) {

      // get octant size
      const real_t octSize = pmesh->getSize(iOct);

      const real_t dx = octSize/bx;
      const real_t dy = octSize/by;
      const real_t dz = octSize/bz;

      // coordinates of the lower left corner
      const real_t x0 = pmesh->getNode(iOct, 0)[IX];
      const real_t y0 = pmesh->getNode(iOct, 0)[IY];
      const real_t z0 = pmesh->getNode(iOct, 0)[IZ];

      // Iterating on all cells
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        KOKKOS_LAMBDA(const int32_t index) {
	  coord_t iCoord;
          uint32_t& ix = iCoord[IX];
          uint32_t& iy = iCoord[IY];
          uint32_t& iz = iCoord[IZ];                    

          if (twoD) {
            iCoord = index_to_coord<2>(index,blockSizes);
          } else {
            iCoord = index_to_coord<3>(index,blockSizes);
          }

          real_t x = x0 + ix*dx + dx/2;
          real_t y = y0 + iy*dy + dy/2;
          real_t z = z0 + iz*dy + dz/2;

	  /**
	   * Compute and assign the variables of the problem here !
	   **/
	  Udata_h(index, fm[ID], iOct) = ...;
	  Udata_h(index, fm[IU], iOct) = ...;
	  Udata_h(index, fm[IV], iOct) = ...;
	  Udata_h(index, fm[IP], iOct) = ...;
	  if (threeD)
	    Udata_h(index, fm[IW], iOct) = ...;
	});
      
      iOct += nbTeams;
    } // end while iOct
  } // end operator()
  	  
  //! Mesh of the simulation
  std::shared_ptr<AMRmesh> pmesh;

  //! Global parameters of the simulation
  HydroParams params;

  //! Specific parameters
  XXXXXParams xxParams;

  //! Field manager
  id2index_t fm;

  //! Block sizes array
  blockSize_t blockSizes;     

  //! Data array on host
  DataArrayBlockHost Udata_h;	  
};

/**
 * Functor to initialize the XXXXX problem
 *
 * This functor takes as input an unrefined mesh and performs initial refinement
 *
 * The parameter structures XXXXXParams (if necessary) should be located in 
 * src/shared/problems
 **/
class InitXXXXXRefineFunctor {
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;
  InitXXXXXRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
			 HydroParams              params,
			 XXXXXParams              xxParams,
			 uint8_t                  level_refine)
    : pmesh(pmesh), params(params), xxParams(xxParams), level_refine(level_refine){};

  // Applying the functor to the mesh
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap                configMap,
		    HydroParams              params,
		    uint8_t level_refine) {
    XXXXXParams xxParams(configMap);

    // iterate functor for refinement
    InitXXXXXRefineFunctor functor(pmesh, params, xxParams, level_refine);

    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());	
    Kokkos::parallel_for("dyablo::muscl_block::InitXXXXXRefineFunctor", policy, functor);
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
      const real_t z = center[2];
      
      double cellSize2 = pmesh->getSize(iOct)*0.75;
      
      bool should_refine = false;

      /**
       * Here, do some calculation, then set should_refine to true when necessary
       **/
      if (should_refine)
        pmesh->setMarker(iOct, 1);
    } // end if level == level_refine
  } // end operator()

  //! The mesh to refine
  std::shared_ptr<AMRmesh> pmesh;

  //! Global simulation parameters
  HydroParams params;

  //! Specific parameters
  XXXXXParams xxParams;

  //! Level to refine
  uint8_t level_refine;
};

} // end namespace muscl_block
} // end namespace dyablo

#endif
