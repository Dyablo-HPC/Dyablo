#ifndef HYDRO_INIT_FUNCTORS_H_
#define HYDRO_INIT_FUNCTORS_H_

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
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * Initial conditions is refined near interface separating the 4 four quadrants.
 *
 */
class InitFourQuadrantFunctor {

public:
  InitFourQuadrantFunctor(const AMRmesh &mesh,
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
    mesh(mesh), params(params), fm(fm), Udata(Udata),
    U0(U0), U1(U1), U2(U2), U3(U3), xt(xt), yt(yt)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(const AMRmesh &mesh,
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
    
    InitFourQuadrantFunctor functor(mesh, params, fm, Udata,
				    configNumber,
				    U0, U1, U2, U3, xt, yt);
    Kokkos::parallel_for(mesh.getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {

    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double,3> center = mesh.getCenter(i);
    
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

  const AMRmesh &mesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  HydroState2d U0, U1, U2, U3;
  real_t       xt, yt;
  
}; // InitFourQuadrantFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // HYDRO_INIT_FUNCTORS_H_
