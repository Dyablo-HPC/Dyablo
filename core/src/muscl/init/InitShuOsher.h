/**
 * \file InitShuOsher.h
 * \author Pierre Kestener
 *
 * ref article : https://doi.org/10.1016/0021-9991(89)90222-2
 * Efficient implementation of essentially non-oscillatory shock-capturing
 * schemes, II, C.-W. Shu and S. Osher, Journal of Computational Physics,
 * Volume 83, Issue 1, July 1989, Pages 32-78.
 *
 * For simplicity, the original problem is rescale in the unit square.
 */
#ifndef HYDRO_INIT_SHU_OSHER_H_
#define HYDRO_INIT_SHU_OSHER_H_

#include <limits> // for std::numeric_limits
#include <memory> // for std::shared_ptr

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/amr/AMRmesh.h"

namespace dyablo { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve Shu-Osher problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 * 
 *
 * Initial conditions is refined near the central discontinuity.
 *
 */
class InitShuOsherDataFunctor {
  
public:
  InitShuOsherDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                          HydroParams   params,
                          id2index_t    fm,
                          DataArray     Udata) :
    pmesh(pmesh), params(params),
    fm(fm), Udata(Udata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams   params,
                    ConfigMap     configMap,
		    id2index_t    fm,
                    DataArray     Udata)
  {
    // data init functor
    InitShuOsherDataFunctor functor(pmesh, params, fm, Udata);

    Kokkos::parallel_for("dyablo::muscl::InitShuOsherDataFunctor", 
                         Kokkos::RangePolicy<Kokkos::OpenMP>(0, pmesh->getNumOctants()), functor);
  }

  void operator()(const size_t& i) const
  {
        
    const real_t gamma0            = params.settings.gamma0;

    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double,3> center = pmesh->getCenter(i);
    
    const real_t x = center[0];
    //const real_t y = center[1];
    //const real_t z = center[2];

    const real_t rho1 = 27.0/7;
    const real_t rho2 = (1.0+sin(50*x-25)/5);

    const real_t p1 = 31.0/3;
    const real_t p2 = 1.0;

    const real_t v1 = 4*sqrt(35.0)/9;
    const real_t v2 = 0;

    if (x < 0.1) {
      Udata(i    , fm[ID]) = rho1;
      Udata(i    , fm[IP]) = p1/(gamma0-1.0) + 0.5*rho1*v1*v1; 
      Udata(i    , fm[IU]) = rho1 * v1;
      Udata(i    , fm[IV]) = 0.0;
    } else {
      Udata(i    , fm[ID]) = rho2;
      Udata(i    , fm[IP]) = p2/(gamma0-1.0) + 0.5*rho2*v2*v2; 
      Udata(i    , fm[IU]) = rho2 * v2;
      Udata(i    , fm[IV]) = 0.0;
    }

    if (params.dimType == THREE_D)
      Udata(i, fm[IW]) = 0.0;
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  
}; // InitShuOsherDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve Shu-Osher problem.
 *
 * This functor only performs mesh refinement based on geometric 
 * information, no user data init.
 * 
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitShuOsherDataFunctor
 *
 */
// class InitShuOsherRefineFunctor {
  
// public:
//   InitShuOsherRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
//                             HydroParams  params,
//                             int         level_refine) :
//     pmesh(pmesh), params(params),
//     level_refine(level_refine)
//   {};
  
//   // static method which does it all: create and execute functor
//   static void apply(std::shared_ptr<AMRmesh> pmesh,
//                     ConfigMap     configMap,
// 		    HydroParams   params,
// 		    int           level_refine)
//   {
//     // iterate functor for refinement
//     InitShuOsherRefineFunctor functor(pmesh, params, 
//                                       level_refine);
//     Kokkos::parallel_for("dyablo::muscl::InitShuOsherRefineFunctor", 
//                          pmesh->getNumOctants(), functor);
    
//   }
  
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const size_t& i) const
//   {

//     // get cell level
//     uint8_t level = pmesh->getLevel(i);
    
//     // only look at level - 1
//     if (level == level_refine) {

//       // get cell center coordinate in the unit domain
//       // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
//       std::array<double,3> center = pmesh->getCenter(i);
      
//       const real_t x = center[0];
//       //const real_t y = center[1];
//       //const real_t z = center[2];

//       double cellSize2 = pmesh->getSize(i)*0.75;
      
//       bool should_refine = false;

//       if ( (x+cellSize2>= 0.5 and x-cellSize2 < 0.5) )
// 	should_refine = true;
      
//       if (should_refine)
// 	pmesh->setMarker(i, 1);

//     } // end if level == level_refine
    
//   } // end operator ()

//   std::shared_ptr<AMRmesh> pmesh;
//   HydroParams    params;
//   int            level_refine;
  
// }; // InitShuOsherRefineFunctor

} // namespace muscl

} // namespace dyablo

#endif // HYDRO_INIT_SHU_OSHER_H_
