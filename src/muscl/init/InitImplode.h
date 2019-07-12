/**
 * \file InitImplode.h
 * \author Pierre Kestener
 */
#ifndef HYDRO_INIT_IMPLODE_H_
#define HYDRO_INIT_IMPLODE_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/ImplodeParams.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace dyablo { 
namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve implode problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 * 
 *
 * Initial conditions is refined near strong density gradients.
 *
 */
class InitImplodeDataFunctor {

public:
  InitImplodeDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                         HydroParams   params,
                         ImplodeParams iParams,
                         id2index_t    fm,
                         DataArray     Udata) :
    pmesh(pmesh), params(params), iParams(iParams),
    fm(fm), Udata(Udata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams   params,
                    ConfigMap     configMap,
		    id2index_t    fm,
                    DataArray     Udata)
  {
    ImplodeParams implodeParams = ImplodeParams(configMap);
    
    // data init functor
    InitImplodeDataFunctor functor(pmesh, params, implodeParams, fm, Udata);
    
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {
    
    const real_t xmin = params.xmin;
    const real_t xmax = params.xmax;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    // implode problem parameters
    // outer parameters
    const real_t rho_out = iParams.rho_out;
    const real_t p_out = iParams.p_out;
    const real_t u_out = iParams.u_out;
    const real_t v_out = iParams.v_out;
    const real_t w_out = iParams.w_out;

    // inner parameters
    const real_t rho_in = iParams.rho_in;
    const real_t p_in = iParams.p_in;
    const real_t u_in = iParams.u_in;
    const real_t v_in = iParams.v_in;
    const real_t w_in = iParams.w_in;
    
    const int shape = iParams.shape;
     
    const real_t gamma0 = params.settings.gamma0;

    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double,3> center = pmesh->getCenter(i);
    
    const real_t x = center[0];
    const real_t y = center[1];
    const real_t z = center[2];

    bool tmp;
    if (shape == 1) {
      if (params.dimType == TWO_D)
        tmp = x+y*y > 0.5 and x+y*y < 1.5;
      else
        tmp = x+y+z > 0.5 and x+y+z < 2.5;
    } else {
      if (params.dimType == TWO_D)
        tmp = x+y > (xmin+xmax)/2. + ymin;
      else
        tmp = x+y+z > (xmin+xmax)/2. + ymin + zmin;
    }

    if (tmp) {
      Udata(i, fm[ID]) = rho_out; 
      Udata(i, fm[IP]) = p_out/(gamma0-1.0) + 
        0.5 * rho_out * (u_out*u_out + v_out*v_out);
      Udata(i, fm[IU]) = u_out;
      Udata(i, fm[IV]) = v_out;
    } else {
      Udata(i, fm[ID]) = rho_in;
      Udata(i, fm[IP]) = p_in/(gamma0-1.0) +
        0.5 * rho_in * (u_in*u_in + v_in*v_in);
      Udata(i, fm[IU]) = u_in;
      Udata(i, fm[IV]) = v_in;
    }
    
    if (params.dimType == THREE_D) { 
      if (tmp) {
        Udata(i, fm[IW]) = w_out;
        Udata(i, fm[IP]) = p_out/(gamma0-1.0) +
          0.5 * rho_out * (u_out*u_out + v_out*v_out + w_out*w_out);
      } else {
        Udata(i, fm[IW]) = w_in;
        Udata(i, fm[IP]) = p_in/(gamma0-1.0) +
          0.5 * rho_in * (u_in*u_in + v_in*v_in + w_in*w_in);
      }
    }
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams   params;
  ImplodeParams iParams;
  id2index_t    fm;
  DataArray     Udata;
  
}; // InitImplodeDataFunctor

} // namespace muscl

} // namespace dyablo

#endif // HYDRO_INIT_IMPLODE_H_
