/**
 * \file MusclBlockGodunovUpdateFunctor.h
 * \author Pierre Kestener
 */
#ifndef MUSCL_BLOCK_GODUNOV_UPDATE_HYDRO_FUNCTOR_H_
#define MUSCL_BLOCK_GODUNOV_UPDATE_HYDRO_FUNCTOR_H_

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"
#include "shared/RiemannSolvers.h"
#include "shared/bc_utils.h"

// utils hydro
#include "shared/utils_hydro.h"

namespace dyablo { 

namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Perform time integration (MUSCL Godunov) update functor.
 *
 * This is tentative of all-in-one functor:
 * - input is Ugroup (containing ghosted block data)
 *
 * We start by computing slopes (along X,Y,Z directions) and
 * store results in shared array.
 *
 * Then we compute fluxes (using Riemann solver) and perform
 * update directly in external array U2.
 * Loop through all cell (sub-)faces:
 *
 */
class MusclBlockGodunovUpdateFunctor {

private:
  using offsets_t = Kokkos::Array<real_t,3>;
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;};

  /**
   * Perform time integration (MUSCL Godunov).
   *
   * \param[in]  pmesh pointer to AMR mesh structure
   * \param[in]  params
   * \param[in]  fm field map
   * \param[in]  Ugroup current time step data (conservative variables)
   * \param[out] U2 next time step data (conservative variables)
   * \param[in]  Qgroup primitive variables
   * \param[in]  time step (as cmputed by CFL condition)
   *
   */
  MusclBlockGodunovUpdateFunctor(std::shared_ptr<AMRmesh> pmesh,
                                 HydroParams    params,
                                 id2index_t     fm,
                                 DataArrayBlock Ugroup,
                                 DataArrayBlock U2,
                                 DataArrayBlock Qgroup,
                                 real_t         dt) :
    pmesh(pmesh),
    params(params),
    fm(fm),
    Ugroup(Ugroup),
    U2(U2),
    Qgroup(Qgroup),
    dt(dt)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap      configMap,
                    HydroParams    params,
		    id2index_t     fm,
		    DataArrayBlock Ugroup,
		    DataArrayBlock U2,
                    DataArrayBlock Qgroup,
                    real_t         dt)
  {

    // instantiate functor
    MusclBlockGodunovUpdateFunctor functor(pmesh, params, fm,
                                           Ugroup, U2,
                                           Qgroup,
                                           dt);

    // create kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("amr","nbTeams",16);
    functor.setNbTeams ( nbTeams_ );
    
    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    // launch computation
    Kokkos::parallel_for("dyablo::muscl::MusclBlockGodunovUpdateFunctor",
                         policy, functor);

  } // apply

  // =======================================================================
  // =======================================================================
  /**
   * a dummy swap device routine.
   */
  KOKKOS_INLINE_FUNCTION
  void swap ( real_t& a, real_t& b ) const {

    real_t tmp(a); a=b; b=tmp;

  } // swap
  
  // =======================================================================
  // =======================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets.
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \note offsets are given in units dx/2, i.e. a vector containing only 1.0 or -1.0
   *
   * \param[in] q primitive variables at cell center
   * \param[in] dqX primitive variables slopes along X
   * \param[in] dqY primitive variables slopes along Y
   * \param[in] offsets identifies where to reconstruct
   * \param[in] dtdx dt divided by dx
   * \param[in] dtdy dt divided by dy
   *
   * \return qr reconstructed state (primitive variables)
   */
  KOKKOS_INLINE_FUNCTION
  HydroState2d reconstruct_state_2d(HydroState2d q, 
                                    HydroState2d dqX, 
                                    HydroState2d dqY, 
                                    offsets_t    offsets,
                                    real_t       dtdx,
                                    real_t       dtdy) const
  {
    const double gamma  = params.settings.gamma0;
    const double smallr = params.settings.smallr;
    
    // retrieve primitive variables in current quadrant
    const real_t r = q[ID];
    const real_t p = q[IP];
    const real_t u = q[IU];
    const real_t v = q[IV];
    const real_t w = 0.0;

    const real_t drx = dqX[ID] * 0.5;
    const real_t dpx = dqX[IP] * 0.5;
    const real_t dux = dqX[IU] * 0.5;
    const real_t dvx = dqX[IV] * 0.5;
    const real_t dwx = 0.0;
    
    const real_t dry = dqY[ID] * 0.5;
    const real_t dpy = dqY[IP] * 0.5;
    const real_t duy = dqY[IU] * 0.5;
    const real_t dvy = dqY[IV] * 0.5;
    const real_t dwy = 0.0;
        
    // source terms (with transverse derivatives)
    const real_t sr0 = (-u*drx-dux*r      )*dtdx + (-v*dry-dvy*r      )*dtdy;
    const real_t su0 = (-u*dux-dpx/r      )*dtdx + (-v*duy            )*dtdy;
    const real_t sv0 = (-u*dvx            )*dtdx + (-v*dvy-dpy/r      )*dtdy;
    const real_t sp0 = (-u*dpx-dux*gamma*p)*dtdx + (-v*dpy-dvy*gamma*p)*dtdy;
    
    // reconstruct state on interface
    HydroState2d qr;
    
    qr[ID] = r + sr0 + offsets[IX] * drx + offsets[IY] * dry;
    qr[IP] = p + sp0 + offsets[IX] * dpx + offsets[IY] * dpy;
    qr[IU] = u + su0 + offsets[IX] * dux + offsets[IY] * duy ;
    qr[IV] = v + sv0 + offsets[IX] * dvx + offsets[IY] * dvy ;
    qr[ID] = fmax(smallr, qr[ID]);

    return qr;
    
  } // reconstruct_state_2d

  // =======================================================================
  // =======================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets (3d version).
   *
   * \sa reconstruct_state_2d
   */
  KOKKOS_INLINE_FUNCTION
  HydroState3d reconstruct_state_3d(HydroState3d q,  
                                    HydroState3d dqX,
                                    HydroState3d dqY,                                   
                                    HydroState3d dqZ,
                                    offsets_t offsets,
                                    real_t dtdx,
                                    real_t dtdy,
                                    real_t dtdz) const
  {
    const double gamma  = params.settings.gamma0;
    const double smallr = params.settings.smallr;
    
    // retrieve primitive variables in current quadrant
    const real_t r = q[ID];
    const real_t p = q[IP];
    const real_t u = q[IU];
    const real_t v = q[IV];
    const real_t w = q[IW];

    // retrieve variations = dx * slopes
    const real_t drx = dqX[ID] * 0.5;
    const real_t dpx = dqX[IP] * 0.5; 
    const real_t dux = dqX[IU] * 0.5;
    const real_t dvx = dqX[IV] * 0.5;
    const real_t dwx = dqX[IW] * 0.5;
    
    const real_t dry = dqY[ID] * 0.5;
    const real_t dpy = dqY[IP] * 0.5;
    const real_t duy = dqY[IU] * 0.5;
    const real_t dvy = dqY[IV] * 0.5;
    const real_t dwy = dqY[IW] * 0.5;
    
    const real_t drz = dqZ[ID] * 0.5;
    const real_t dpz = dqZ[IP] * 0.5;
    const real_t duz = dqZ[IU] * 0.5;
    const real_t dvz = dqZ[IV] * 0.5;
    const real_t dwz = dqZ[IW] * 0.5;
    
    // source terms (with transverse derivatives)
    const real_t sr0 = (-u*drx-dux*r      )*dtdx + (-v*dry-dvy*r      )*dtdy + (-w*drz-dwz*r      )*dtdz;
    const real_t su0 = (-u*dux-dpx/r      )*dtdx + (-v*duy            )*dtdy + (-w*duz            )*dtdz;
    const real_t sv0 = (-u*dvx            )*dtdx + (-v*dvy-dpy/r      )*dtdy + (-w*dvz            )*dtdz;
    const real_t sw0 = (-u*dwx            )*dtdx + (-v*dwy            )*dtdy + (-w*dwz-dpz/r      )*dtdz;
    const real_t sp0 = (-u*dpx-dux*gamma*p)*dtdx + (-v*dpy-dvy*gamma*p)*dtdy + (-w*dpz-dwz*gamma*p)*dtdz;

    // reconstruct state on interface
    HydroState3d qr;

    qr[ID] = r + sr0 + offsets[IX] * drx + offsets[IY] * dry + offsets[IZ] * drz ;
    qr[IP] = p + sp0 + offsets[IX] * dpx + offsets[IY] * dpy + offsets[IZ] * dpz ;
    qr[IU] = u + su0 + offsets[IX] * dux + offsets[IY] * duy + offsets[IZ] * duz ;
    qr[IV] = v + sv0 + offsets[IX] * dvx + offsets[IY] * dvy + offsets[IZ] * dvz ;
    qr[IW] = w + sw0 + offsets[IX] * dwx + offsets[IY] * dwy + offsets[IZ] * dwz ;

    qr[ID] = fmax(smallr, qr[ID]);

    return qr;
    
  } // reconstruct_state_3d

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_2d(team_policy_t::member_type member) const 
  {

  } // operator_2d

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_3d(team_policy_t::member_type member) const 
  {

  } // operator_3d

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const
  {
    
    if (this->params.dimType == TWO_D)
      operator_2d(member);
    
    if (this->params.dimType == THREE_D)
      operator_3d(member);
    
  } // operator ()
  
  std::shared_ptr<AMRmesh> pmesh;
  HydroParams    params;
  id2index_t     fm;
  DataArrayBlock Ugroup, U2;
  DataArrayBlock Qgroup;
  real_t         dt;
  
}; // MusclBlockGodunovUpdateFunctor

} // namespace muscl

} // namespace dyablo

#endif // MUSCL_BLOCK_GODUNOV_UPDATE_HYDRO_FUNCTOR_H_
