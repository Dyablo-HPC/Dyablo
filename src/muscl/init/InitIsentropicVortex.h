/**
 * \file InitIsentropic.h
 * \author Pierre Kestener
 */
#ifndef HYDRO_INIT_ISENTROPIC_VORTEX_H_
#define HYDRO_INIT_ISENTROPIC_VORTEX_H_

#include <limits> // for std::numeric_limits

#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/kokkos_shared.h"
#include "shared/problems/IsentropicVortexParams.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace euler_pablo {
namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve the isentropic vortex problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 *
 * Initial conditions is refined near strong density gradients.
 *
 */
class InitIsentropicVortexDataFunctor {

public:
  InitIsentropicVortexDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                                  HydroParams params,
                                  IsentropicVortexParams ivParams,
                                  id2index_t fm, DataArray Udata)
      : pmesh(pmesh), params(params), ivParams(ivParams), fm(fm),
        Udata(Udata){};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh, 
                    HydroParams params,
                    ConfigMap configMap, 
                    id2index_t fm, 
                    DataArray Udata) 
  {
    // isentropic vortex specific parameters
    IsentropicVortexParams ivParams = IsentropicVortexParams(configMap);

    // data init functor
    InitIsentropicVortexDataFunctor functor(pmesh, params, ivParams, fm, Udata);

    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t &i) const {

    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double, 3> center = pmesh->getCenter(i);

    const real_t x = center[0];
    const real_t y = center[1];
    //const real_t z = center[2];

    // fluid specific heat ratio
    const real_t gamma0 = params.settings.gamma0;

    // isentropic vortex parameters

    // ambient flow
    const real_t rho_a = ivParams.rho_a;
    // const real_t p_a   = ivParams.p_a;
    const real_t T_a = ivParams.T_a;
    const real_t u_a = ivParams.u_a;
    const real_t v_a = ivParams.v_a;
    // const real_t w_a   = ivParams.w_a;

    // vortex center
    const real_t vortex_x = ivParams.vortex_x;
    const real_t vortex_y = ivParams.vortex_y;

    const real_t scale = ivParams.scale;

    // relative coordinates versus vortex center
    real_t xp = (x - vortex_x)/scale;
    real_t yp = (y - vortex_y)/scale;
    real_t r = sqrt(xp * xp + yp * yp);

    const real_t beta = ivParams.beta;

    real_t du = -yp * beta / (2 * M_PI) * exp(0.5 * (1.0 - r * r));
    real_t dv = xp * beta / (2 * M_PI) * exp(0.5 * (1.0 - r * r));

    real_t T = T_a - (gamma0 - 1) * beta * beta / (8 * gamma0 * M_PI * M_PI) *
                         exp(1.0 - r * r);
    real_t rho = rho_a * pow(T / T_a, 1.0 / (gamma0 - 1));

    Udata(i, fm[ID]) = rho;
    Udata(i, fm[IU]) = rho * (u_a + du);
    Udata(i, fm[IV]) = rho * (v_a + dv);
    // Udata(i  ,j  , IP) = pow(rho,gamma0)/(gamma0-1.0) +
    Udata(i, fm[IP]) = rho * T / (gamma0 - 1.0) +
                       0.5 * rho * (u_a + du) * (u_a + du) +
                       0.5 * rho * (v_a + dv) * (v_a + dv);

    if (params.dimType == THREE_D)
      Udata(i, fm[IW]) = 0.0;

  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams params;
  IsentropicVortexParams ivParams;
  id2index_t fm;
  DataArray Udata;

}; // InitIsentropicVortexDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve isentropic vortex problem.
 *
 * This functor only performs mesh refinement, no user data init.
 *
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitIsentropicVortexDataFunctor
 *
 */
class InitIsentropicVortexRefineFunctor {

public:
  InitIsentropicVortexRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                                    HydroParams params,
                                    IsentropicVortexParams ivParams,
                                    int level_refine)
      : pmesh(pmesh), params(params), ivParams(ivParams),
        level_refine(level_refine){};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh, ConfigMap configMap,
                    HydroParams params, int level_refine) {
    IsentropicVortexParams ivParams = IsentropicVortexParams(configMap);

    // iterate functor for refinement
    InitIsentropicVortexRefineFunctor functor(pmesh, params, ivParams,
                                              level_refine);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t &i) const {

    // get cell level
    uint8_t level = pmesh->getLevel(i);

    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      //std::array<double, 3> center = pmesh->getCenter(i);

      //const real_t x = center[0];
      //const real_t y = center[1];
      //const real_t z = center[2];

      //double cellSize2 = pmesh->getSize(i) * 0.75;

      bool should_refine = false;

      // TODO
      should_refine = true;

      if (should_refine)
        pmesh->setMarker(i, 1);

    } // end if level == level_refine

  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams params;
  IsentropicVortexParams ivParams;
  int level_refine;

}; // InitIsentropicVortexRefineFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // HYDRO_INIT_ISENTROPIC_VORTEX_H_
