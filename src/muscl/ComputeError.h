/**
 * \file ComputeError.h
 * \author Pierre Kestener
 */
#ifndef MUSCL_COMPUTE_ERROR_FUNCTOR_H_
#define MUSCL_COMPUTE_ERROR_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h> // for cuda math constants, e.g. CUDART_INF
#endif // __CUDA_ARCH__

#include "shared/kokkos_shared.h"
#include "muscl/HydroBaseFunctor.h"


//#include "shared/EulerEquations.h"

namespace euler_pablo {
namespace muscl {

enum norm_type {
  NORM_L1,
  NORM_L2
};

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * compute L1 / L2 error between two data array (solution data).
 *
 */
class Compute_Error_Functor : public HydroBaseFunctor {

public:
  
  /**
   * \param[in] varId identify which variable to reduce (ID, IE, IU, ...)
   */
  Compute_Error_Functor(HydroParams         params,
                        DataArray           Udata1,
                        DataArray           Udata2,
                        int                 varId,
                        norm_type           norm) :
    HydroBaseFunctor(params),
    Udata1(Udata1),
    Udata2(Udata2),
    varId(varId),
    norm(norm)
  {};
  
  // static method which does it all: create and execute functor
  static double apply(HydroParams       params,
                      DataArray         Udata1,
                      DataArray         Udata2,
                      int               varId,
                      norm_type         norm)
  {
    int64_t nbCells = Udata1.extent(0);

    real_t error = 0;
    Compute_Error_Functor functor(params, Udata1, Udata2, varId, norm);
    Kokkos::parallel_reduce(nbCells, functor, error);
    return error;
  }
  
  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (real_t& dst) const
  {
    // The identity under '+' is zero.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
    dst = 0.0;
  } // init

  //! the functor 
  KOKKOS_INLINE_FUNCTION
  void operator()(const int index,
		  real_t &sum) const
  {
      
    // get local conservative variable
    real_t tmp1 = Udata1(index,varId);
    real_t tmp2 = Udata2(index,varId);
    
    if (norm == NORM_L1) {
      sum += fabs(tmp1-tmp2);
    } else {
      sum += (tmp1-tmp2)*(tmp1-tmp2);
    }
    
  } // end operator () - 2d
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile real_t& dst,
	     const volatile real_t& src) const
  {
    // + reduce
    dst += src;
  } // join

  DataArray Udata1;
  DataArray Udata2;
  int varId;
  norm_type norm;

}; // class Compute_Error_Functor

} // namespace muscl
} // namespace euler_pablo

#endif // MUSCL_COMPUTE_ERROR_FUNCTOR_H_
