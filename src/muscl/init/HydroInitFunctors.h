#ifndef HYDRO_INIT_FUNCTORS_H_
#define HYDRO_INIT_FUNCTORS_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/initRiemannConfig2d.h"

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
  InitFourQuadrantFunctor(HydroParams   params,
			  id2index_t    fm,
			  DataArray     Udata,
			  int           configNumber,
			  HydroState2d  U0,
			  HydroState2d  U1,
			  HydroState2d  U2,
			  HydroState2d  U3,
			  real_t        xt,
			  real_t        yt) :
    params(params), fm(fm), Udata(Udata),
    U0(U0), U1(U1), U2(U2), U3(U3), xt(xt), yt(yt)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(HydroParams   params,
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
    
    InitFourQuadrantFunctor functor(params, fm, Udata, configNumber,
				    U0, U1, U2, U3, xt, yt);
    //Kokkos::parallel_for(nbCells, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int& index) const
  {

//     const int isize = params.isize;
//     const int jsize = params.jsize;
//     const int ghostWidth = params.ghostWidth;
    
// #ifdef USE_MPI
//     const int i_mpi = params.myMpiPos[IX];
//     const int j_mpi = params.myMpiPos[IY];
// #else
//     const int i_mpi = 0;
//     const int j_mpi = 0;
// #endif

//     const int nx = params.nx;
//     const int ny = params.ny;

//     const real_t xmin = params.xmin;
//     const real_t ymin = params.ymin;
//     const real_t dx = params.dx;
//     const real_t dy = params.dy;
    
//     int i,j;
//     index2coord(index,i,j,isize,jsize);
    
//     real_t x = xmin + dx/2 + (i+nx*i_mpi-ghostWidth)*dx;
//     real_t y = ymin + dy/2 + (j+ny*j_mpi-ghostWidth)*dy;
    
//     if (x<xt) {
//       if (y<yt) {
// 	// quarter 2
// 	Udata(i  ,j  , ID) = U2[ID];
// 	Udata(i  ,j  , IP) = U2[IP];
// 	Udata(i  ,j  , IU) = U2[IU];
// 	Udata(i  ,j  , IV) = U2[IV];
//       } else {
// 	// quarter 1
// 	Udata(i  ,j  , ID) = U1[ID];
// 	Udata(i  ,j  , IP) = U1[IP];
// 	Udata(i  ,j  , IU) = U1[IU];
// 	Udata(i  ,j  , IV) = U1[IV];
//       }
//     } else {
//       if (y<yt) {
// 	// quarter 3
// 	Udata(i  ,j  , ID) = U3[ID];
// 	Udata(i  ,j  , IP) = U3[IP];
// 	Udata(i  ,j  , IU) = U3[IU];
// 	Udata(i  ,j  , IV) = U3[IV];
//       } else {
// 	// quarter 0
// 	Udata(i  ,j  , ID) = U0[ID];
// 	Udata(i  ,j  , IP) = U0[IP];
// 	Udata(i  ,j  , IU) = U0[IU];
// 	Udata(i  ,j  , IV) = U0[IV];
//       }
//     }
    
  } // end operator ()

  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  HydroState2d U0, U1, U2, U3;
  real_t       xt, yt;
  
}; // InitFourQuadrantFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // HYDRO_INIT_FUNCTORS_H_
