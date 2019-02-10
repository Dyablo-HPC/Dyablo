#ifndef HYDRO_INIT_FOUR_QUANDRANT_H_
#define HYDRO_INIT_FOUR_QUANDRANT_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/initRiemannConfig2d.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace euler_pablo { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve a four quadrant 2D Riemann
 * problem.
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

public:
  InitFourQuadrantDataFunctor(std::shared_ptr<AMRmesh> pmesh,
			      HydroParams   params,
			      id2index_t    fm,
			      DataArray     Udata,
			      int           configNumber,
			      HydroState2d  U0,
			      HydroState2d  U1,
			      HydroState2d  U2,
			      HydroState2d  U3,
			      real_t        xt,
			      real_t        yt) :
    pmesh(pmesh), params(params), fm(fm), Udata(Udata),
    U0(U0), U1(U1), U2(U2), U3(U3), xt(xt), yt(yt)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams   params,
		    id2index_t    fm,
                    DataArray     Udata,
		    int           configNumber,
		    HydroState2d  U0,
		    HydroState2d  U1,
		    HydroState2d  U2,
		    HydroState2d  U3,
		    real_t        xt,
		    real_t        yt)
  {
    
    // iterate functor for refinement
    
    InitFourQuadrantDataFunctor functor(pmesh, params, fm, Udata,
					configNumber,
					U0, U1, U2, U3, xt, yt);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {
    
    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double,3> center = pmesh->getCenter(i);
    
    const real_t x = center[0];
    const real_t y = center[1];
    
    if (x<xt) {
      if (y<yt) {
	// quarter 2
	Udata(i    , fm[ID]) = U2[ID];
	Udata(i    , fm[IP]) = U2[IP];
	Udata(i    , fm[IU]) = U2[IU];
	Udata(i    , fm[IV]) = U2[IV];
      } else {
	// quarter 1
	Udata(i    , fm[ID]) = U1[ID];
	Udata(i    , fm[IP]) = U1[IP];
	Udata(i    , fm[IU]) = U1[IU];
	Udata(i    , fm[IV]) = U1[IV];
      }
    } else {
      if (y<yt) {
	// quarter 3
	Udata(i    , fm[ID]) = U3[ID];
	Udata(i    , fm[IP]) = U3[IP];
	Udata(i    , fm[IU]) = U3[IU];
	Udata(i    , fm[IV]) = U3[IV];
      } else {
	// quarter 0
	Udata(i    , fm[ID]) = U0[ID];
	Udata(i    , fm[IP]) = U0[IP];
	Udata(i    , fm[IU]) = U0[IU];
	Udata(i    , fm[IV]) = U0[IV];
      }
    }
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  HydroState2d U0, U1, U2, U3;
  real_t       xt, yt;
  
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
  InitFourQuadrantRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
				HydroParams   params,
				int           level_refine,
				real_t        xt,
				real_t        yt) :
    pmesh(pmesh), params(params),
    level_refine(level_refine),
    xt(xt), yt(yt)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams   params,
		    int           level_refine,
		    real_t        xt,
		    real_t        yt)
  {
    
    // iterate functor for refinement
    InitFourQuadrantRefineFunctor functor(pmesh, params, level_refine, xt, yt);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
    
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {

    //constexpr double eps = 0.005;
    
    // get cell level
    uint8_t level = pmesh->getLevel(i);
    
    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double,3> center = pmesh->getCenter(i);
      
      const real_t x = center[0];
      const real_t y = center[1];
      
      double cellSize2 = pmesh->getSize(i)*0.75;
      
      bool should_refine = false;

      if ((x+cellSize2>= xt and x-cellSize2 < xt) or
	  (x-cellSize2<= xt and x+cellSize2 > xt) or
	  (y+cellSize2>= yt and y-cellSize2 < yt) or
	  (y-cellSize2<= yt and y+cellSize2 > yt)   )
	should_refine = true;
      
      if (should_refine)
	pmesh->setMarker(i, 1);

    } // end if level == level_refine
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams    params;
  int            level_refine;
  real_t         xt, yt;
  
}; // InitFourQuadrantRefineFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // HYDRO_INIT_FOUR_QUANDRANT_H_
