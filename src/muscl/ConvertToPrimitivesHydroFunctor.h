/**
 * \file ConvertToPrimitivesHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef CONVERT_TO_PRIMITIVES_HYDRO_FUNCTOR_H_
#define CONVERT_TO_PRIMITIVES_HYDRO_FUNCTOR_H_

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// base class
#include "muscl/HydroBaseFunctor.h"

namespace dyablo { namespace muscl {

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
    Kokkos::parallel_for("dyablo::muscl::ConvertToPrimitivesHydroFunctor",
                         pmesh->getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator_2d(const size_t& i) const
  {
    
    HydroState2d uLoc; // conservative variables in current cell
    HydroState2d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    
    // get local conservative variable
    uLoc[ID] = Udata(i,fm[ID]);
    uLoc[IP] = Udata(i,fm[IP]);
    uLoc[IU] = Udata(i,fm[IU]);
    uLoc[IV] = Udata(i,fm[IV]);
    
    // get primitive variables in current cell
    computePrimitives(uLoc, &c, qLoc);
    
    // copy q state in q global
    Qdata(i,fm[ID]) = qLoc[ID];
    Qdata(i,fm[IP]) = qLoc[IP];
    Qdata(i,fm[IU]) = qLoc[IU];
    Qdata(i,fm[IV]) = qLoc[IV];

  } // operator_2d

  KOKKOS_INLINE_FUNCTION
  void operator_3d(const size_t& i) const
  {
    
    HydroState3d uLoc; // conservative variables in current cell
    HydroState3d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    
    // get local conservative variable
    uLoc[ID] = Udata(i,fm[ID]);
    uLoc[IP] = Udata(i,fm[IP]);
    uLoc[IU] = Udata(i,fm[IU]);
    uLoc[IV] = Udata(i,fm[IV]);
    uLoc[IW] = Udata(i,fm[IW]);
    
    // get primitive variables in current cell
    computePrimitives(uLoc, &c, qLoc);
    
    // copy q state in q global
    Qdata(i,fm[ID]) = qLoc[ID];
    Qdata(i,fm[IP]) = qLoc[IP];
    Qdata(i,fm[IU]) = qLoc[IU];
    Qdata(i,fm[IV]) = qLoc[IV];
    Qdata(i,fm[IW]) = qLoc[IW];

  } // operator_3d

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {
    
    if (this->params.dimType == TWO_D)
      operator_2d(i);
    
    if (this->params.dimType == THREE_D)
      operator_3d(i);
    
  } // operator ()
  
  std::shared_ptr<AMRmesh> pmesh;
  id2index_t   fm;
  DataArray    Udata;
  DataArray    Qdata;

}; // ConvertToPrimitivesHydroFunctor

} // namespace muscl

} // namespace dyablo

#endif // CONVERT_TO_PRIMITIVES_HYDRO_FUNCTOR_H_
