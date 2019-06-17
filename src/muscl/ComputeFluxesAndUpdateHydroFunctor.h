/**
 * \file ComputeFluxesAndUpdateHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef COMPUTE_FLUXES_AND_UPDATE_HYDRO_FUNCTOR_H_
#define COMPUTE_FLUXES_AND_UPDATE_HYDRO_FUNCTOR_H_

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
 * Compute Fluxes and Update functor.
 *
 * Loop through all cell (sub-)faces:
 * - compute fluxes: need first to reconstruct data on both sides
 *   of an interface, then call a Riemann solver
 * - update only current cell
 *
 */
class ComputeFluxesAndUpdateHydroFunctor : public HydroBaseFunctor {
  
public:
  /**
   * Reconstruct gradients
   *
   * \param[in] params
   * \param[in] Udata conservative variables - needed ???
   * \param[in] Qdata primitive variables
   */
  ComputeFluxesAndUpdateHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
                                     HydroParams params,
                                     id2index_t    fm,
                                     DataArray Data_in,
                                     DataArray Data_out,
                                     DataArray Qdata,
                                     DataArray SlopeX,
                                     DataArray SlopeY,
                                     DataArray SlopeZ,
                                     real_t    dt) :
    HydroBaseFunctor(params),
    pmesh(pmesh),
    fm(fm),
    Data_in(Data_in),
    Data_out(Data_out),
    Qdata(Qdata),
    SlopeX(SlopeX),
    SlopeY(SlopeY),
    SlopeZ(SlopeZ),
    dt(dt)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
		    DataArray Data_in,
		    DataArray Data_out,
                    DataArray Qdata,
		    DataArray SlopeX,
		    DataArray SlopeY,
		    DataArray SlopeZ,
                    real_t    dt)
  {
    ComputeFluxesAndUpdateHydroFunctor functor(pmesh, params, fm, 
                                               Data_in, Data_out,
                                               Qdata,
                                               SlopeX,SlopeY,SlopeZ,
                                               dt);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator_2d(const size_t i) const 
  {
        
  } // operator_2d

  KOKKOS_INLINE_FUNCTION
  void operator_3d(const size_t i) const 
  {
  
  } // operator_3d

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const
  {
    
    if (this->params.dimType == TWO_D)
      operator_2d(i);
    
    if (this->params.dimType == THREE_D)
      operator_3d(i);
    
  } // operator ()
  
  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Data_in, Data_out;
  DataArray    Qdata;
  DataArray    SlopeX, SlopeY, SlopeZ;
  real_t       dt;
  
}; // ComputeFluxesAndUpdateHydroFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // COMPUTE_FLUXES_AND_UPDATE_HYDRO_FUNCTOR_H_
