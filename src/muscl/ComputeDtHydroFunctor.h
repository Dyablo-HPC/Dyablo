#ifndef COMPUTE_DT_HYDRO_FUNCTOR_H_
#define COMPUTE_DT_HYDRO_FUNCTOR_H_

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
 */
class ComputeDtHydroFunctor {

public:
  ComputeDtHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
			HydroParams   params,
			id2index_t    fm,
			DataArray     Udata) :
    pmesh(pmesh), params(params), fm(fm), Udata(Udata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams   params,
		    id2index_t    fm,
                    DataArray     Udata,
		    double       &invDt)
  {
    
    // iterate functor for refinement
    
    ComputeDtHydroFunctor functor(pmesh, params, fm, Udata);
    Kokkos::parallel_reduce(pmesh->getNumOctants(), functor, invDt);
  }
  
  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (real_t& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
  } // init

  KOKKOS_INLINE_FUNCTION
  void operator_2d()(const size_t& i, real_t &invDt) const
  {

    // 2D version
    HydroState2d uLoc; // conservative variables in current cell
    HydroState2d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    real_t vx, vy;
    
    // get local conservative variable
    uLoc[ID] = Udata(i,fm[ID]);
    uLoc[IP] = Udata(i,fm[IP]);
    uLoc[IU] = Udata(i,fm[IU]);
    uLoc[IV] = Udata(i,fm[IV]);
    
    // get primitive variables in current cell
    computePrimitives(uLoc, &c, qLoc);
    vx = c+FABS(qLoc[IU]);
    vy = c+FABS(qLoc[IV]);
    
    invDt = FMAX(invDt, vx/dx + vy/dy);

  } // operator_2d ()

  KOKKOS_INLINE_FUNCTION
  void operator_3d()(const size_t& i, real_t &invDt) const
  {
  } // operator_3d

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i, real_t &invDt) const
  {

    if (params.dimType == TWO_D)
      operator_2d(i,invDt);

    if (params.dimType == THREE_D)
      operator_3d(i,invDt);
    
  }
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile real_t& dst,
	     const volatile real_t& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  
}; // ComputeDtHydroFunctor

#endif // COMPUTE_DT_HYDRO_FUNCTOR_H_
