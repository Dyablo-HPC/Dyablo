#ifndef CONVERT_TO_PRIMITIVES_HYDRO_FUNCTOR_H_
#define CONVERT_TO_PRIMITIVES_HYDRO_FUNCTOR_H_

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
 * Convert conservative variables to primitive variables functor.
 *
 */
class ConvertToPrimitivesHydroFunctor : public HydroBaseFunctor {

public:
  /**
   * Convert conservative variables to primitive ones using equation of state.
   *
   * \param[in] params
   * \param[in] Udata conservative variables
   * \param[out] Qdata primitive variables
   */
  ConvertToPrimitivesHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
				  HydroParams params,
				  id2index_t    fm,
				  DataArray Udata,
				  DataArray Qdata) :
    HydroBaseFunctor(params),
    pmesh(pmesh), fm(fm), Udata(Udata), Qdata(Qdata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
		    DataArray Udata,
                    DataArray Qdata)
  {
    ConvertToPrimitivesHydroFunctor functor(pmesh, params, fm, Udata, Qdata);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {

    // TODO !!!
  }

  
  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  DataArray    Qdata;

}; // ConvertToPrimitivesHydroFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // CONVERT_TO_PRIMITIVES_HYDRO_FUNCTOR_H_
