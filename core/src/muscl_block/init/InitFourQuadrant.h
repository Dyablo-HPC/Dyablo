/**
 * \file InitFourQuadrant.h
 * \author Maxime Delorme
 * \author Pierre Kestener
 **/
#ifndef MUSCL_BLOCK_HYDRO_INIT_FOUR_QUANDRANT_H_
#define MUSCL_BLOCK_HYDRO_INIT_FOUR_QUANDRANT_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/initRiemannConfig2d.h"
#include "shared/problems/FourQuadrantParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace dyablo { namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve a four quadrant 2D Riemann
 * problem in the Block mode.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 * 
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * Initial conditions is refined near interface separating the 4 
 * four quadrants.
 *
 */
class InitFourQuadrantDataFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams
  
public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;
  
  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  InitFourQuadrantDataFunctor(std::shared_ptr<AMRmesh> pmesh,
			      HydroParams         params,
			      FourQuadrantParams  fqParams,
			      id2index_t          fm,
			      blockSize_t         blockSizes,
			      DataArrayBlockHost  Udata_h) :
    pmesh(pmesh), params(params), fqParams(fqParams),
    fm(fm), blockSizes(blockSizes), Udata_h(Udata_h)
  {
    // initializing the four states
    getRiemannConfig2d(fqParams.configNumber, U0, U1, U2, U3);
    primToCons_2D(U0, params.settings.gamma0);
    primToCons_2D(U1, params.settings.gamma0);
    primToCons_2D(U2, params.settings.gamma0);
    primToCons_2D(U3, params.settings.gamma0);
  };
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams    params,
                    ConfigMap      configMap,
		    id2index_t     fm,
                    blockSize_t    blockSizes,
                    DataArrayBlockHost Udata_h)
  {
    FourQuadrantParams fqParams = FourQuadrantParams(configMap);
    
    // data init functor
    InitFourQuadrantDataFunctor functor(pmesh, params, fqParams, fm, blockSizes, Udata_h);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("init","nbTeams",16);
    functor.setNbTeams ( nbTeams_  );

    // perform initialization on host
    team_policy_t policy (Kokkos::OpenMP(),
                          nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("dyablo::muscl_block::InitFourQuadrantDataFunctor",
                         policy, functor);
  }
  
  //KOKKOS_INLINE_FUNCTION
  inline
  void operator()(thread_t member) const
  {
    uint32_t iOct = member.league_rank();
    
    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    while (iOct <  pmesh->getNumOctants() )
    {
      // get octant size
      const real_t octSize = pmesh->getSize(iOct);

      const real_t dx = octSize/bx;
      const real_t dy = octSize/by;

      const real_t x0 = pmesh->getCoordinates(iOct)[IX];
      const real_t y0 = pmesh->getCoordinates(iOct)[IY];

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        KOKKOS_LAMBDA(const int32_t index) {

	  coord_t iCoord;
          uint32_t& ix = iCoord[IX];
          uint32_t& iy = iCoord[IY];
          uint32_t& iz = iCoord[IZ];                    

          if (params.dimType == TWO_D) {
            iCoord = index_to_coord<2>(index,blockSizes);
          } else {
            iCoord = index_to_coord<3>(index,blockSizes);
          }
	  
	  // Compute position
	  const real_t x = x0 + ix*dx + dx/2;
          const real_t y = y0 + iy*dy + dy/2;

	  // And assign values
	  if (x<fqParams.xt) {
	    if (y<fqParams.yt) {
	      // quarter 2
	      Udata_h(index, fm[ID], iOct) = U2[ID];
	      Udata_h(index, fm[IP], iOct) = U2[IP];
	      Udata_h(index, fm[IU], iOct) = U2[IU];
	      Udata_h(index, fm[IV], iOct) = U2[IV];
	    } else {
	      // quarter 1
	      Udata_h(index, fm[ID], iOct) = U1[ID];
	      Udata_h(index, fm[IP], iOct) = U1[IP];
	      Udata_h(index, fm[IU], iOct) = U1[IU];
	      Udata_h(index, fm[IV], iOct) = U1[IV];
	    }
	  } else {
	    if (y<fqParams.yt) {
	      // quarter 3
	      Udata_h(index, fm[ID], iOct) = U3[ID];
	      Udata_h(index, fm[IP], iOct) = U3[IP];
	      Udata_h(index, fm[IU], iOct) = U3[IU];
	      Udata_h(index, fm[IV], iOct) = U3[IV];
	    } else {
	      // quarter 0
	      Udata_h(index, fm[ID], iOct) = U0[ID];
	      Udata_h(index, fm[IP], iOct) = U0[IP];
	      Udata_h(index, fm[IU], iOct) = U0[IU];
	      Udata_h(index, fm[IV], iOct) = U0[IV];
	    }
	  }
	  if (params.dimType == THREE_D)
	    Udata_h(index, fm[IW], iOct) = 0.0;
	});
      iOct += nbTeams;
    } // end while iOct
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  
  HydroParams        params;
  FourQuadrantParams fqParams;
  id2index_t         fm;
  blockSize_t        blockSizes;
  DataArrayBlockHost Udata_h;

  HydroState2d U0, U1, U2, U3;
}; // InitFourQuadrantDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve a four quadrant 2D Riemann
 * problem.
 *
 * This functor only performs mesh refinement, no user data init.
 * 
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * Initial conditions is refined near interface separating the 4 
 * four quadrants.
 *
 * \sa InitFourQuadrantDataFunctor
 *
 */
class InitFourQuadrantRefineFunctor {
  
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;
  InitFourQuadrantRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
				HydroParams        params,
				FourQuadrantParams fqParams,
				int                level_refine) :
    pmesh(pmesh), params(params), fqParams(fqParams),
    level_refine(level_refine)
  {}
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap     configMap,
		    HydroParams   params,
		    int           level_refine)
  {
    FourQuadrantParams fqParams = FourQuadrantParams(configMap);
    
    // iterate functor for refinement
    InitFourQuadrantRefineFunctor functor(pmesh, params, fqParams, level_refine);

    
    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());
    Kokkos::parallel_for("dyablo::muscl_block::InitFourQuadrantRefineFunctor",
                         policy, functor);
  }
  
  void operator()(const size_t& iOct) const
  {
    // get cell level
    uint8_t level = pmesh->getLevel(iOct);
    
    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double,3> center = pmesh->getCenter(iOct);
      
      const real_t x = center[0];
      const real_t y = center[1];
      
      double cellSize2 = pmesh->getSize(iOct)*0.75;
      
      bool should_refine = false;

      const real_t xt = fqParams.xt;
      const real_t yt = fqParams.yt;
      
      if ((x+cellSize2>= xt and x-cellSize2 < xt) or
	  (x-cellSize2<= xt and x+cellSize2 > xt) or
	  (y+cellSize2>= yt and y-cellSize2 < yt) or
	  (y-cellSize2<= yt and y+cellSize2 > yt)   )
	should_refine = true;
      
      if (should_refine)
	pmesh->setMarker(iOct, 1);

    } // end if level == level_refine
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams        params;
  FourQuadrantParams fqParams;
  int                level_refine;
  
}; // InitFourQuadrantRefineFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_HYDRO_INIT_FOUR_QUANDRANT_H_
