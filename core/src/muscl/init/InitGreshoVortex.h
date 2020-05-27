/**
 * \file InitGreshoVortex.h
 * \author Pierre Kestener
 */
#ifndef HYDRO_INIT_GRESHO_VORTEX_H_
#define HYDRO_INIT_GRESHO_VORTEX_H_

#include <limits> // for std::numeric_limits

#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/kokkos_shared.h"
#include "shared/problems/GreshoVortexParams.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace dyablo {
namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve the gresho vortex problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 *
 * Initial conditions is refined near strong density gradients.
 *
 */

class InitGreshoVortexDataFunctor {

public:
  InitGreshoVortexDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                              HydroParams params, GreshoVortexParams gvParams,
                              id2index_t fm, DataArray Udata)
      : pmesh(pmesh), params(params), gvParams(gvParams), fm(fm),
        Udata(Udata){};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh, HydroParams params,
                    ConfigMap configMap, id2index_t fm, DataArray Udata) {
    // gresho vortex specific parameters
    GreshoVortexParams gvParams = GreshoVortexParams(configMap);

    // data init functor
    InitGreshoVortexDataFunctor functor(pmesh, params, gvParams, fm, Udata);

    Kokkos::parallel_for("dyablo::muscl::InitGreshoVortexDataFunctor", pmesh->getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t &i) const {

    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double, 3> center = pmesh->getCenter(i);

    real_t x = center[0];
    real_t y = center[1];
    // real_t z = center[2];

    x -= 0.5;
    y -= 0.5;

    // fluid specific heat ratio
    const real_t gamma0 = params.settings.gamma0;

    // gresho vortex parameters

    const real_t rho0  = gvParams.rho0;
    const real_t Ma    = gvParams.Ma;

    const real_t p0 = rho0 / (gamma0 * Ma * Ma);

    real_t r = sqrt(x*x+y*y);
    real_t theta = atan2(y,x);
    
    // polar coordinate
    real_t cosT = cos(theta);
    real_t sinT = sin(theta);
    
    real_t uphi, p;
    
    if ( r < 0.2 ) {
      
      uphi = 5*r;
      p    = p0 + 25/2.0*r*r;
      
    } else if ( r < 0.4 ) {
      
      uphi = 2 - 5 * r;
      p    = p0 + 25/2.0*r*r + 4*(1-5*r-log(0.2)+log(r));
      
    } else {
      
      uphi = 0;
      p    = p0-2+4*log(2.0);
      
    }
    
    Udata(i, fm[ID]) = rho0;
    Udata(i, fm[IU]) = rho0 * (-sinT * uphi);
    Udata(i, fm[IV]) = rho0 * ( cosT * uphi);
    Udata(i, fm[IP]) = p/(gamma0-1.0) +
      0.5*(Udata(i,fm[IU])*Udata(i,fm[IU]) +
           Udata(i,fm[IV])*Udata(i,fm[IV]))/Udata(i,fm[ID]);;
    
    if (params.dimType == THREE_D)
      Udata(i, fm[IW]) = 0.0;

  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams params;
  GreshoVortexParams gvParams;
  id2index_t fm;
  DataArray Udata;

}; // InitGreshoVortexDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve gresho vortex problem.
 *
 * This functor only performs mesh refinement, no user data init.
 *
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitGreshoVortexDataFunctor
 *
 */
class InitGreshoVortexRefineFunctor {

public:
  InitGreshoVortexRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                                HydroParams params, GreshoVortexParams gvParams,
                                int level_refine)
      : pmesh(pmesh), params(params), gvParams(gvParams),
        level_refine(level_refine){};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh, ConfigMap configMap,
                    HydroParams params, int level_refine) 
  {

    GreshoVortexParams gvParams = GreshoVortexParams(configMap);

    // iterate functor for refinement
    InitGreshoVortexRefineFunctor functor(pmesh, params, gvParams,
                                          level_refine);
    Kokkos::parallel_for("dyablo::muscl::InitGreshoVortexRefineFunctor", pmesh->getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t &i) const {

    // get cell level
    uint8_t level = pmesh->getLevel(i);

    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double, 3> center = pmesh->getCenter(i);

      real_t x = center[0];
      real_t y = center[1];
      real_t z = center[2];

      // retrieve coordinates to domain center
      x -= 0.5;
      y -= 0.5;
      z -= 0.5;

      double cellSize2 = pmesh->getSize(i) * 0.75;

      bool should_refine = false;

      // refine near r=0.2
      real_t radius = 0.2;

      real_t d2 = x*x + y*y;

      if (params.dimType == THREE_D)
        d2 += z*z;

      real_t d = sqrt(d2);

      if ( fabs(d - radius) < cellSize2 )
        should_refine = true;
      
      if ( d > 0.15 and d < 0.25 )
        should_refine = true;

      if (should_refine)
        pmesh->setMarker(i, 1);

    } // end if level == level_refine

  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams params;
  GreshoVortexParams gvParams;
  int level_refine;

}; // InitGreshoVortexRefineFunctor

} // namespace muscl

} // namespace dyablo

#endif // HYDRO_INIT_GRESHO_VORTEX_H_
