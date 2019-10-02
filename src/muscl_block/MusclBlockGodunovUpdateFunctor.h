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

// utils block
#include "muscl_block/utils_block.h"

namespace dyablo { 

namespace muscl_block {

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
 * Loop through all cell (sub-)faces.
 *
 * \note This functor actually assumes ghost width is 2.
 * if you really need something else (larger than 2), please
 * refactor.
 *
 * \todo routines like reconstruct_state_2d/3d could probably be
 * moved outside to alleviate this class.
 *
 */
class MusclBlockGodunovUpdateFunctor {

private:
  using offsets_t = Kokkos::Array<real_t,3>;
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  // scratch memory aliases
  using shared_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  using shared_2d_t  = Kokkos::View<real_t**, shared_space, Kokkos::MemoryUnmanaged>;

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
                                 blockSize_t    blockSizes,
                                 uint32_t       ghostWidth,
                                 uint32_t       nbOcts,
                                 uint32_t       nbOctsPerGroup,
                                 uint32_t       iGroup,
                                 DataArrayBlock Ugroup,
                                 DataArrayBlock U2,
                                 DataArrayBlock Qgroup,
                                 real_t         dt) :
    pmesh(pmesh),
    params(params),
    fm(fm),
    blockSizes(blockSizes),
    ghostWidth(ghostWidth),
    nbOcts(nbOcts),
    nbOctsPerGroup(nbOctsPerGroup),
    iGroup(iGroup),
    Ugroup(Ugroup),
    U2(U2),
    Qgroup(Qgroup),
    dt(dt)
  {

    bx_g = blockSizes[IX] + 2 * (ghostWidth);
    by_g = blockSizes[IY] + 2 * (ghostWidth);
    bz_g = blockSizes[IZ] + 2 * (ghostWidth);

    // here we remove 1 to ghostWidth, only the inner
    // part need to compute limited slopes
    bx1  = blockSizes[IX] + 2 * (ghostWidth-1);
    by1  = blockSizes[IY] + 2 * (ghostWidth-1);
    bz1  = blockSizes[IZ] + 2 * (ghostWidth-1);
    
    nbCellsPerBlock = params.dimType == TWO_D ? 
      bx1 * by1 :
      bx1 * by1 * bz1;

  };
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap      configMap,
                    HydroParams    params,
		    id2index_t     fm,
                    blockSize_t    blockSizes,
                    uint32_t       ghostWidth,
                    uint32_t       nbOcts,
                    uint32_t       nbOctsPerGroup,
                    uint32_t       iGroup,
		    DataArrayBlock Ugroup,
		    DataArrayBlock U2,
                    DataArrayBlock Qgroup,
                    real_t         dt)
  {

    // instantiate functor
    MusclBlockGodunovUpdateFunctor functor(pmesh, params, fm,
                                           blockSizes, ghostWidth, 
                                           nbOcts, nbOctsPerGroup, iGroup,
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

  // ====================================================================
  // ====================================================================
  /**
   * Get conservative variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template<class HydroState>
  KOKKOS_INLINE_FUNCTION
  HydroState get_cons_variables(uint32_t index, 
                                uint32_t iOct_local) const
  {

    HydroState q;

    q[ID] = Ugroup(index, fm[ID], iOct_local);
    q[IP] = Ugroup(index, fm[IP], iOct_local);
    q[IU] = Ugroup(index, fm[IU], iOct_local);
    q[IV] = Ugroup(index, fm[IV], iOct_local);
    if (std::is_same<HydroState,HydroState3d>::value)
      q[IW] = Ugroup(index, fm[IW], iOct_local);

    return q;

  } // get_cons_variables

  // ====================================================================
  // ====================================================================
  /**
   * Get primitive variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template<class HydroState>
  KOKKOS_INLINE_FUNCTION
  HydroState get_prim_variables(uint32_t index, 
                                uint32_t iOct_local) const
  {

    HydroState q;

    q[ID] = Qgroup(index, fm[ID], iOct_local);
    q[IP] = Qgroup(index, fm[IP], iOct_local);
    q[IU] = Qgroup(index, fm[IU], iOct_local);
    q[IV] = Qgroup(index, fm[IV], iOct_local);
    if (std::is_same<HydroState,HydroState3d>::value)
      q[IW] = Qgroup(index, fm[IW], iOct_local);

    return q;

  } // get_prim_variables

  // ====================================================================
  // ====================================================================
  /**
   * Compute primitive variables slopes (dq) for one component from q and its neighbors.
   * 
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in] q scalar value in current cell
   * \param[in] qPlus scalar value in right neighbor 
   * \param[in] qMinus scalar value in left neighbor
   *
   * \return dq limited slope (scalar)
   */
  KOKKOS_INLINE_FUNCTION
  real_t slope_unsplit_scalar(real_t q, 
                              real_t qPlus,
                              real_t qMinus) const
  {
    const real_t slope_type = params.settings.slope_type;

    real_t dq = 0;

    if (slope_type == 1 or slope_type == 2) {

      // slopes in first coordinate direction
      const real_t dlft = slope_type * (q - qMinus);
      const real_t drgt = slope_type * (qPlus - q);
      const real_t dcen = HALF_F * (qPlus - qMinus);
      const real_t dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      const real_t slop = fmin(FABS(dlft), FABS(drgt));
      real_t dlim = slop;
      if ((dlft * drgt) <= ZERO_F)
        dlim = ZERO_F;
      dq = dsgn * fmin(dlim, FABS(dcen));
    }

    return dq;

  } // slope_unsplit_scalar

  // ====================================================================
  // ====================================================================
  /**
   * Compute primitive variables slopes (dq) for one component from q and its neighbors.
   * 
   * Only slope_type 1 and 2 are supported.
   *
   * \param[in] ic index (ghosted block) to current cell
   * \param[in] ip index (ghosted block) to next cell
   * \param[in] im index (ghosted block) to previous block
   * \param[in] ivar identifies which variables to read
   * \param[in] iOct_local identifies octant inside current group of octants
   *
   * \return dq limited slope (scalar)
   */
  KOKKOS_INLINE_FUNCTION
  real_t slope_unsplit_scalar(uint32_t ic, 
                              uint32_t ip,
                              uint32_t im,
                              uint32_t ivar,
                              uint32_t iOct_local) const
  {
    const real_t slope_type = params.settings.slope_type;

    real_t dq = 0;
    
    if (slope_type == 1 or slope_type == 2) {
      
      const real_t q      = Qgroup(ic, ivar, iOct_local);
      const real_t qPlus  = Qgroup(ip, ivar, iOct_local);
      const real_t qMinus = Qgroup(im, ivar, iOct_local);
      
      // slopes in first coordinate direction
      const real_t dlft = slope_type * (q - qMinus);
      const real_t drgt = slope_type * (qPlus - q);
      const real_t dcen = HALF_F * (qPlus - qMinus);
      const real_t dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      const real_t slop = fmin(FABS(dlft), FABS(drgt));
      real_t dlim = slop;
      if ((dlft * drgt) <= ZERO_F)
        dlim = ZERO_F;
      dq = dsgn * fmin(dlim, FABS(dcen));
    
    }

    return dq;

  } // slope_unsplit_scalar

  // ====================================================================
  // ====================================================================
  /**
   * Compute slope (vector value using minmod limiter).
   */
  template<class HydroState>
  KOKKOS_INLINE_FUNCTION
  HydroState slope_unsplit_hydro(const HydroState& q,
                                 const HydroState& qPlus,
                                 const HydroState& qMinus) const
  {

    const real_t slope_type = params.settings.slope_type;

    HydroState dq;

    dq[IX] = slope_unsplit_scalar( q[ID], qPlus[ID], qMinus[ID] );
    dq[IP] = slope_unsplit_scalar( q[IP], qPlus[IP], qMinus[IP] );
    dq[IU] = slope_unsplit_scalar( q[IU], qPlus[IU], qMinus[IU] );
    dq[IV] = slope_unsplit_scalar( q[IV], qPlus[IV], qMinus[IV] );
    if (std::is_same<HydroState,HydroState3d>::value)
      dq[IW] = slope_unsplit_scalar( q[IW], qPlus[IW], qMinus[IW] );

    return dq;

  } // slope_unsplit_hydro

  // ====================================================================
  // ====================================================================
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
  HydroState2d reconstruct_state_2d(const HydroState2d& q,
                                    const HydroState2d& dqX, 
                                    const HydroState2d& dqY, 
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

  // ====================================================================
  // ====================================================================
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
  HydroState2d reconstruct_state_2d(const HydroState2d& q,
                                    uint32_t     index,
                                    shared_2d_t  slopesX, 
                                    shared_2d_t  slopesY, 
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
    //const real_t w = 0.0;

    const real_t drx = slopesX(index,fm[ID]) * 0.5;
    const real_t dpx = slopesX(index,fm[IP]) * 0.5;
    const real_t dux = slopesX(index,fm[IU]) * 0.5;
    const real_t dvx = slopesX(index,fm[IV]) * 0.5;
    //const real_t dwx = 0.0;
    
    const real_t dry = slopesY(index,fm[ID]) * 0.5;
    const real_t dpy = slopesY(index,fm[IP]) * 0.5;
    const real_t duy = slopesY(index,fm[IU]) * 0.5;
    const real_t dvy = slopesY(index,fm[IV]) * 0.5;
    //const real_t dwy = 0.0;
        
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

  // ====================================================================
  // ====================================================================
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

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_2d(team_policy_t::member_type member) const 
  {

    const int nbvar = 2;

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;
    
    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();
    
    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    // Allocate a shared array for the team to computes slopes
    shared_2d_t slopesX(member.team_shmem(), nbCellsPerBlock, nbvar);
    shared_2d_t slopesY(member.team_shmem(), nbCellsPerBlock, nbvar);

    const real_t dtdx = 1.0;/* TODO */
    const real_t dtdy = 1.0;/* TODO */

    const uint32_t& bx = blockSizes[IX];

    while (iOct < iOctNextGroup and iOct < nbOcts)
    {

      // step 1 : compute limited slopes
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock),
        KOKKOS_LAMBDA(const int32_t index) {

          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j
          const int j = index / bx1;
          const int i = index - j*bx1;

          // corresponding index in the ghosted block
          // i -> i+1
          // j -> j+1
          const uint32_t ib = (i+1) + bx_g * (j+1);
          
          // neighbor along x axis
          uint32_t ibp1 = ib + 1;
          uint32_t ibm1 = ib - 1;

          slopesX(ib,fm[ID]) = slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);
          slopesX(ib,fm[IP]) = slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);
          slopesX(ib,fm[IU]) = slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);
          slopesX(ib,fm[IV]) = slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);

          // neighbor along y axis
          ibp1 = ib + bx_g;
          ibm1 = ib - bx_g;

          slopesY(ib,fm[ID]) = slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);
          slopesY(ib,fm[IP]) = slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);
          slopesY(ib,fm[IU]) = slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);
          slopesY(ib,fm[IV]) = slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);

          // DEBUG : write into Ugroup
          //Ugroup(ib,fm[ID],iOct_local) = slopesX(ib,fm[ID]);

        }); // end TeamVectorRange

      // step 2 : reconstruct states on cells face and update
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock),
        KOKKOS_LAMBDA(const int32_t index) {

          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j
          const int j = index / bx1;
          const int i = index - j*bx1;

          // corresponding index in the full ghosted block
          // i -> i+1
          // j -> j+1
          const uint32_t ig = (i+1) + bx_g * (j+1);

          // the following condition makes sure we stay inside
          // the inner block
          if (i > 0 and i < bx1 - 1 and 
              j > 0 and j < by1 - 1) {
            // get current location primitive variables state
            HydroState2d qprim = get_prim_variables<HydroState2d>(ig, iOct_local);

            // fluxes will be accumulated in qcons
            HydroState2d qcons = get_cons_variables<HydroState2d>(ig, iOct_local);

            /*
             * compute from left face along x dir
             */
            {
              // step 1 : reconstruct state in the left neighbor

              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig-1, iOct_local);

              // 
              offsets_t offsets = {1.0, 0.0, 0.0};

              // reconstruct "left" state
              HydroState2d qL = reconstruct_state_2d(
                  qprim_n, index-1, slopesX, slopesY, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {-1.0, 0.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(
                qprim, index, slopesX, slopesY, offsets, dtdx, dtdy);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              // step 4 : accumulate flux in current cell
              // qcons += flux*dtdx; // TODO
            }

            /*
             * compute flux from right face along x dir
             */
            {
              // step 1 : reconstruct state in the left neighbor

              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig+1, iOct_local);

              // 
              offsets_t offsets = {-1.0, 0.0, 0.0};

              // reconstruct "right" state
              HydroState2d qR = reconstruct_state_2d(
                  qprim_n, index+1, slopesX, slopesY, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {1.0, 0.0, 0.0};

              HydroState2d qL = reconstruct_state_2d(
                qprim, index, slopesX, slopesY, offsets, dtdx, dtdy);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              // step 4 : accumulate flux in current cell
              // qcons += flux*dtdx; // TODO
            }

            /*
             * compute flux from left face along y dir
             */
            {
              // step 1 : reconstruct state in the left neighbor
              
              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig-bx_g, iOct_local);
              
              // 
              offsets_t offsets = {0.0, 1.0, 0.0};
              
              // reconstruct "left" state
              HydroState2d qL = reconstruct_state_2d(
                  qprim_n, index-bx1, slopesX, slopesY, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {0.0, -1.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(
                qprim, index, slopesX, slopesY, offsets, dtdx, dtdy);

              // swap IU / IV
              my_swap(qL[IU], qL[IV]);
              my_swap(qR[IU], qR[IV]);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              my_swap(flux[IU], flux[IV]);

              // step 4 : accumulate flux in current cell
              // qcons += flux*dtdx; // TODO
            }

            /*
             * compute flux from right face along y dir
             */
            { 
              // step 1 : reconstruct state in the left neighbor
              
              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig+bx_g, iOct_local);
              
              // 
              offsets_t offsets = {0.0, -1.0, 0.0};
              
              // reconstruct "left" state
              HydroState2d qL = reconstruct_state_2d(
                  qprim_n, index+bx1, slopesX, slopesY, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {0.0, 1.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(
                qprim, index, slopesX, slopesY, offsets, dtdx, dtdy);

              // swap IU / IV
              my_swap(qL[IU], qL[IV]);
              my_swap(qR[IU], qR[IV]);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              my_swap(flux[IU], flux[IV]);

              // step 4 : accumulate flux in current cell
              // qcons += flux*dtdx; // TODO

            }

            // lastly update conservative variable in U2
            uint32_t index_non_ghosted = (i-1) + bx * (j-1);

            U2(index_non_ghosted, fm[ID], iOct) = qcons[ID];
            U2(index_non_ghosted, fm[IP], iOct) = qcons[IP];
            U2(index_non_ghosted, fm[IU], iOct) = qcons[IU];
            U2(index_non_ghosted, fm[IV], iOct) = qcons[IV];

          } // end if inside inner block

        }); // end TeamVectorRange


      iOct       += nbTeams;
      iOct_local += nbTeams;

    } // end while iOct < nbOct

  } // operator_2d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_3d(team_policy_t::member_type member) const 
  {

  } // operator_3d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const
  {
    
    if (this->params.dimType == TWO_D)
      operator_2d(member);
    
    else if (this->params.dimType == THREE_D)
      operator_3d(member);
    
  } // operator ()

  //! bitpit/PABLO amr mesh object
  std::shared_ptr<AMRmesh> pmesh;
  
  //! general parameters
  HydroParams    params;

  //! field manager
  id2index_t     fm;

  //! block sizes (no ghost)
  blockSize_t blockSizes;

  //! blockSizes with ghost
  uint32_t bx_g;
  uint32_t by_g;
  uint32_t bz_g;

  //! blockSizes with ghost (minus 1)
  uint32_t bx1;
  uint32_t by1;
  uint32_t bz1;

  //! ghost width
  uint32_t ghostWidth;

  //! total number of octants in current MPI process
  uint32_t nbOcts;

  //! number of octant per group
  uint32_t nbOctsPerGroup;

  //! number of cells per block
  uint32_t nbCellsPerBlock;

  //! integer which identifies a group of octants
  uint32_t iGroup;

  //! user data for the ith group of octants
  DataArrayBlock Ugroup;

  //! user data for the entire mesh at the end of time step
  DataArrayBlock U2;

  //! user data (primitive variables) for the ith group of octants
  DataArrayBlock Qgroup;

  //! time step
  real_t         dt;
  
}; // MusclBlockGodunovUpdateFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_GODUNOV_UPDATE_HYDRO_FUNCTOR_H_
