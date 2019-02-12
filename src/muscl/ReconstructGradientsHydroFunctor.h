#ifndef RECONSTRUCT_GRADIENTS_HYDRO_FUNCTOR_H_
#define RECONSTRUCT_GRADIENTS_HYDRO_FUNCTOR_H_

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// base class
#include "muscl/HydroBaseFunctor.h"

namespace euler_pablo { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Reconstruct gradients functor.
 *
 * Equivalent to slope limiter in a regular cartesian grid code.
 *
 */
class ReconstructGradientsHydroFunctor : public HydroBaseFunctor {
  
public:
  /**
   * Reconstruct gradients
   *
   * \param[in] params
   * \param[in] Udata conservative variables - needed ???
   * \param[in] Qdata primitive variables
   */
  ReconstructGradientsHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
				   HydroParams params,
				   id2index_t    fm,
				   DataArray Udata,
				   DataArray Qdata,
				   DataArray SlopeX,
				   DataArray SlopeY,
				   DataArray SlopeZ) :
    HydroBaseFunctor(params),
    pmesh(pmesh), fm(fm), Udata(Udata), Qdata(Qdata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
		    DataArray Udata,
                    DataArray Qdata,
		    DataArray SlopeX,
		    DataArray SlopeY,
		    DataArray SlopeZ)
  {
    ReconstructGradientsHydroFunctor functor(pmesh, params, fm, Udata, Qdata);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }
  
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {
    
    // TODO
    
  } // operator ()
  
  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  DataArray    Qdata;
  DataArray    SlopeX, SlopeY, SlopeZ;
  
}; // ReconstructGradientsHydroFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // RECONSTRUCT_GRADIENTS_HYDRO_FUNCTOR_H_
