/**
 * \file InitSod.h
 * \author Pierre Kestener
 */
#ifndef HYDRO_INIT_SOD_H_
#define HYDRO_INIT_SOD_H_

#include <limits> // for std::numeric_limits
#include <memory> // for std::shared_ptr

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace euler_pablo { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve sod problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 * 
 *
 * Initial conditions is refined near the central discontinuity.
 *
 */
class InitSodDataFunctor {

public:
  InitSodDataFunctor(std::shared_ptr<AMRmesh> pmesh,
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
    InitSodDataFunctor functor(pmesh, params, fm, Udata);

    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {
        
    const real_t gamma0            = params.settings.gamma0;

    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double,3> center = pmesh->getCenter(i);
    
    const real_t x = center[0];
    //const real_t y = center[1];
    //const real_t z = center[2];
    
    if (x < 0.5) {
      Udata(i    , fm[ID]) = 1.0;
      Udata(i    , fm[IP]) = 1.0/(gamma0-1.0); 
      Udata(i    , fm[IU]) = 0.0;
      Udata(i    , fm[IV]) = 0.0;
    } else {
      Udata(i    , fm[ID]) = 0.125;
      Udata(i    , fm[IP]) = 0.1/(gamma0-1.0); 
      Udata(i    , fm[IU]) = 0.0;
      Udata(i    , fm[IV]) = 0.0;
    }

    if (params.dimType == THREE_D)
      Udata(i, fm[IW]) = 0.0;
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  
}; // InitSodDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve sod problem.
 *
 * This functor only performs mesh refinement, no user data init.
 * 
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitSodDataFunctor
 *
 */
class InitSodRefineFunctor {
  
public:
  InitSodRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                       HydroParams  params,
                       int         level_refine) :
    pmesh(pmesh), params(params),
    level_refine(level_refine)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap     configMap,
		    HydroParams   params,
		    int           level_refine)
  {
    // iterate functor for refinement
    InitSodRefineFunctor functor(pmesh, params, 
                                 level_refine);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
    
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {

    // get cell level
    uint8_t level = pmesh->getLevel(i);
    
    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double,3> center = pmesh->getCenter(i);
      
      const real_t x = center[0];
      //const real_t y = center[1];
      //const real_t z = center[2];

      double cellSize2 = pmesh->getSize(i)*0.75;
      
      bool should_refine = false;

      if ( (x+cellSize2>= 0.5 and x-cellSize2 < 0.5) )
	should_refine = true;
      
      if (should_refine)
	pmesh->setMarker(i, 1);

    } // end if level == level_refine
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams    params;
  int            level_refine;
  
}; // InitSodRefineFunctor

class SolverHydroMuscl;
void init_sod(SolverHydroMuscl *psolver);

} // namespace muscl

} // namespace euler_pablo

#endif // HYDRO_INIT_SOD_H_
