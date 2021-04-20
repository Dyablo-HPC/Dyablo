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
#include "shared/amr/AMRmesh.h"
//#include "shared/RiemannSolvers.h"
//#include "shared/bc_utils.h"

// utils hydro
#include "shared/utils_hydro.h"

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
class UpdateRSSTHydroFunctor {

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
    pmesh(pmesh),
    params(params),
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
    Kokkos::parallel_for("dyablo::muscl::UpdateRSSTHydroFunctor", pmesh->getNumOctants(), functor);
  }

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_2d(const size_t i) const 
  {

    const int nbvar = params.nbvar;

    // speed of sound reducer (must be larger >= 1)
    const real_t ksi = params.rsst_ksi;

    // compute prefactor 1-1/ksi^2
    const real_t ksi2 = 1.0 - 1.0/ksi/ksi;

    // current cell center state (primitive variables)
    HydroState2d qprim;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qprim[ivar] = Qdata(i,fm[ivar]);

    // current cell conservative variable state
    HydroState2d qcons;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qcons[ivar] = Data_in(i,fm[ivar]);

    // current cell pressure and speed of sound
    real_t pressure, cs;
    compute_Pressure_and_SpeedOfSound(qcons, pressure, cs, params);
    const real_t cs2 = cs*cs;

    // read flux
    real_t delta_rho    = Fluxes(i,fm[ID]);
    real_t delta_rhov_x = Fluxes(i,fm[IU]);
    real_t delta_rhov_y = Fluxes(i,fm[IV]);
    real_t delta_e_tot  = Fluxes(i,fm[IE]);

    // compute V^2
    const real_t V2 = qprim[IU]*qprim[IU] + qprim[IV]*qprim[IV];

    // pressure correction (ideal equation of state)
    const real_t gamma0 = params.settings.gamma0;
    const real_t delta_P = (gamma0-1) * ( (0.5*V2)*delta_rho 
                                    - qprim[IU] * delta_rhov_x
                                    - qprim[IV] * delta_rhov_y
                                    + delta_e_tot );

    const real_t delta_P2 = ksi2 / cs2 * delta_P;

    // compute corrected fluxes
    delta_rho    -= delta_P2;
    delta_rhov_x -= delta_P2 * qprim[IU];
    delta_rhov_y -= delta_P2 * qprim[IV];
    delta_e_tot  -= delta_P2 * (qprim[IP]+qcons[IE])/qprim[ID]; 

    // now update
    Data_out(i,fm[ID]) = qcons[ID] + delta_rho;
    Data_out(i,fm[IU]) = qcons[IU] + delta_rhov_x;
    Data_out(i,fm[IV]) = qcons[IV] + delta_rhov_y;
    Data_out(i,fm[IE]) = qcons[IE] + delta_e_tot;

  } // operator_2d

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_3d(const size_t i) const {

    const int nbvar = params.nbvar;

    // speed of sound reducer (must be larger >= 1)
    const real_t ksi = params.rsst_ksi;

    // compute prefactor 1-1/ksi^2
    const real_t ksi2 = 1.0 - 1.0/ksi/ksi;

    // current cell center state (primitive variables)
    HydroState3d qprim;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qprim[ivar] = Qdata(i,fm[ivar]);

    // current cell conservative variable state
    HydroState3d qcons;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qcons[ivar] = Data_in(i,fm[ivar]);

    // current cell pressure and speed of sound
    real_t pressure, cs;
    compute_Pressure_and_SpeedOfSound(qcons, pressure, cs, params);
    const real_t cs2 = cs*cs;

    // read flux
    real_t delta_rho    = Fluxes(i,fm[ID]);
    real_t delta_rhov_x = Fluxes(i,fm[IU]);
    real_t delta_rhov_y = Fluxes(i,fm[IV]);
    real_t delta_rhov_z = Fluxes(i,fm[IW]);
    real_t delta_e_tot  = Fluxes(i,fm[IE]);

    // compute V^2
    const real_t V2 = qprim[IU]*qprim[IU] + qprim[IV]*qprim[IV] + qprim[IW]*qprim[IW];

    // pressure correction (ideal equation of state)
    const real_t gamma0 = params.settings.gamma0;
    const real_t delta_P = (gamma0-1) * ( (0.5*V2)*delta_rho 
                                    - qprim[IU] * delta_rhov_x
                                    - qprim[IV] * delta_rhov_y
                                    - qprim[IW] * delta_rhov_z
                                    + delta_e_tot );

    const real_t delta_P2 = ksi2 / cs2 * delta_P;

    // compute corrected fluxes
    delta_rho    -= delta_P2;
    delta_rhov_x -= delta_P2 * qprim[IU];
    delta_rhov_y -= delta_P2 * qprim[IV];
    delta_rhov_z -= delta_P2 * qprim[IW];
    delta_e_tot  -= delta_P2 * (qprim[IP]+qcons[IE])/qprim[ID]; 

    // now update
    Data_out(i,fm[ID]) = qcons[ID] + delta_rho;
    Data_out(i,fm[IU]) = qcons[IU] + delta_rhov_x;
    Data_out(i,fm[IV]) = qcons[IV] + delta_rhov_y;
    Data_out(i,fm[IW]) = qcons[IW] + delta_rhov_z;
    Data_out(i,fm[IE]) = qcons[IE] + delta_e_tot;

  } // operator_3d

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
  HydroParams  params;
  id2index_t   fm;
  DataArray    Data_in, Data_out;
  DataArray    Qdata;
  DataArray    Fluxes;

}; // class UpdateRSSTHydroFunctor

} // namespace muscl

} // namespace dyablo

#endif // UPDATE_RSST_HYDRO_FUNCTOR_H_
