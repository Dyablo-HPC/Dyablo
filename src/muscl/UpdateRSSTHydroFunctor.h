/**
 * \file UpdateRSSTHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef UPDATE_RSST_HYDRO_FUNCTOR_H_
#define UPDATE_RSST_HYDRO_FUNCTOR_H_

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"
//#include "shared/RiemannSolvers.h"
//#include "shared/bc_utils.h"

// base class
#include "muscl/HydroBaseFunctor.h"

namespace dyablo { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Update hydrodynamics when RSST (Reduced Speed of Sound Technique) is actiated.
 *
 * Here we are using a special RSST, modified for flow with large density variation,
 * and presented in article
 * https://arxiv.org/abs/1812.04135

 * Loop through all cell :
 * - read fluxes as computed by ComputeFluxesAndUpdateHydroFunctor
 * - compute RSST source terms from equation (20) and (21)
 * - perform the actual update
 *
 * \sa functor ComputeFluxesAndUpdateHydroFunctor
 *
 */
class UpdateRSSTHydroFunctor : public HydroBaseFunctor {

public:
  /**
   * Constructor for UpdateRSSTHydroFunctor.
   *
   * \param[in]  pmesh pointer to AMR mesh structure
   * \param[in]  params
   * \param[in]  fm field map
   * \param[in]  Data_in current time step data (conservative variables)
   * \param[out] Data_out next time step data (conservative variables)
   * \param[in]  Qdata primitive variables
   * \param[in]  Fluxes fluxes (time update as computed by a regular existing solver)
   *
   */
  UpdateRSSTHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
                         HydroParams params,
                         id2index_t    fm,
                         DataArray Data_in,
                         DataArray Data_out,
                         DataArray Qdata,
                         DataArray Fluxes) :
    HydroBaseFunctor(params),
    pmesh(pmesh),
    fm(fm),
    Data_in(Data_in),
    Data_out(Data_out),
    Qdata(Qdata),
    Fluxes(Fluxes)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
		    DataArray Data_in,
		    DataArray Data_out,
                    DataArray Qdata,
		    DataArray Fluxes)
  {
    UpdateRSSTHydroFunctor functor(pmesh, params, fm, 
                                   Data_in, Data_out,
                                   Qdata, Fluxes);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_2d(const size_t i) const {    
  }

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_3d(const size_t i) const {
  }

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const
  {
    
    if (this->params.dimType == TWO_D)
      operator_2d(i);
    
    if (this->params.dimType == THREE_D)
      operator_3d(i);
    
  } // operator ()

  std::shared_ptr<AMRmesh> pmesh;
  id2index_t   fm;
  DataArray    Data_in, Data_out;
  DataArray    Qdata;
  DataArray    Fluxes;

}; // class UpdateRSSTHydroFunctor

} // namespace muscl

} // namespace dyablo

#endif // UPDATE_RSST_HYDRO_FUNCTOR_H_
