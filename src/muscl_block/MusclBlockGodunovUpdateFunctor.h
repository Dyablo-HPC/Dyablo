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
 * This functor contains actually two stages (parallel loops)
 *  - one for slopes computations
 *  - one for flux / update computations
 *
 * input data is Ugroup (containing ghosted block data)
 *
 * We start by computing slopes (along X,Y,Z directions) and
 * store results in locally allocated array (sized upon Qgroup).
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

  //! Tag structure for the 2 stages
  struct Slopes {}; // stage 1
  struct Fluxes {}; // stage 2

  using team_policy1_t = Kokkos::TeamPolicy<Slopes, Kokkos::IndexType<int32_t>>;
  using thread1_t = team_policy1_t::member_type;

  using team_policy2_t = Kokkos::TeamPolicy<Fluxes, Kokkos::IndexType<int32_t>>;
  using thread2_t = team_policy2_t::member_type;

  // scratch memory aliases
  //using shared_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  //using shared_2d_t  = Kokkos::View<real_t**, shared_space, Kokkos::MemoryUnmanaged>;

public:
  using index_t = int32_t;
  
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
				 DataArrayBlock U,
				 DataArrayBlock U_ghost,
                                 DataArrayBlock U2,
                                 DataArrayBlock Qgroup,
                                 FlagArrayBlock Interface_flags,
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
    U(U),
    U_ghost(U_ghost),
    U2(U2),
    Qgroup(Qgroup),
    Interface_flags(Interface_flags),
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
    
    nbCellsPerBlock1 = params.dimType == TWO_D ? 
      bx1 * by1 :
      bx1 * by1 * bz1;

    // memory allocation for slopes arrays
    SlopesX = DataArrayBlock("SlopesX", 
                             Qgroup.extent(0), 
                             Qgroup.extent(1), 
                             Qgroup.extent(2));
    SlopesY = DataArrayBlock("SlopesY", 
                             Qgroup.extent(0), 
                             Qgroup.extent(1), 
                             Qgroup.extent(2));
    if (params.dimType == THREE_D)
      SlopesZ = DataArrayBlock("SlopesZ", 
                               Qgroup.extent(0), 
                               Qgroup.extent(1), 
                               Qgroup.extent(2));
      
  }; // constructor
  
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
		    DataArrayBlock U,
		    DataArrayBlock U_ghost,
		    DataArrayBlock U2,
                    DataArrayBlock Qgroup,
                    FlagArrayBlock Interface_flags,
                    real_t         dt)
    {

    // instantiate functor
    MusclBlockGodunovUpdateFunctor functor(pmesh, params, fm,
                                           blockSizes, ghostWidth, 
                                           nbOcts, nbOctsPerGroup, iGroup,
                                           Ugroup, U, U_ghost, U2,
                                           Qgroup,
                                           Interface_flags,
                                           dt);

    uint32_t nbTeams_ = configMap.getInteger("amr","nbTeams",16);
    functor.setNbTeams ( nbTeams_ );
    
    // create kokkos execution policy for slopes computation
    team_policy1_t policy1 (nbTeams_,
                            Kokkos::AUTO() /* team size */);

    // launch computation
    Kokkos::parallel_for("dyablo::muscl::MusclBlockGodunovUpdateFunctor - Slopes",
                         policy1, functor);

    // create kokkos execution policy for fluxes/update computation
    team_policy2_t policy2 (nbTeams_,
                            Kokkos::AUTO() /* team size */);

    // launch computation
    Kokkos::parallel_for("dyablo::muscl::MusclBlockGodunovUpdateFunctor - Fluxes/Update",
                         policy2, functor);

  } // apply

  // ====================================================================
  // ====================================================================
  /** 
   * Extracts non conformal neighbor states
   * 
   * \param[in]  iOct index of the current oct in the entire mesh
   * \param[in]  i index on dir_x of the current cell in the original block (non ghosted)
   * \param[in]  j index on dir_y of the current cell in the original block (non ghosted) 
   * \param[in]  dir direction along which neighbors should be retrieved
   * \param[in]  face face along which neighbors should be retrieved
   * \param[out] qm first neighboring state to be retrieved
   * \param[out] qp second neighboring state
   */
  KOKKOS_INLINE_FUNCTION
  void get_non_conformal_neighbors_2d(uint32_t iOct,
				      uint32_t i,
				      uint32_t j,
				      DIR_ID  dir,
				      FACE_ID face,
				      HydroState2d &qm,
				      HydroState2d &qp) const {

    uint8_t codim = 1;
    
    const uint32_t &bx = blockSizes[IX];
    const uint32_t &by = blockSizes[IY];
    
    uint8_t iface = face + 2*dir;

    std::vector<uint32_t> neigh;
    std::vector<bool> is_ghost;

    pmesh->findNeighbours(iOct, iface, codim, neigh, is_ghost);
    
    uint32_t ii, jj; // Coords of the first neighbour
    uint8_t iNeigh = 0;
    
    if (dir == DIR_X) {
      ii = bx-i-1;
      jj = j*2;
    }
    else {
      jj = by-j-1;
      ii = i*2;
    }

    // We go through the two sub-cells
    for (uint8_t ip=0; ip < 2; ++ip) {
      if (dir == DIR_X) {
	jj += ip;
	if (jj >= by) {
	  iNeigh = 1;
	  jj -= by;
	}
      }
      else {
	ii += ip;
	if (ii >= bx) {
	  iNeigh = 1;
	  ii -= bx;
	}
      }

      //uint32_t index = i + bx * j;
      /*
      std::cout << "Cell corresponding to (" << i << "; " << j
		<< "); For Dir=" << (int)dir << " and Face=" << (int)face
		<< " with ip=" << (int)ip
		<< " is (" << ii << "; " << jj << "; " << (int)iNeigh << ")" << std::endl;*/
      
      uint32_t index_border = ii + bx * jj;
      HydroState2d &q = (ip == 0 ? qm : qp);

      //std::cout << " SUMMARY: " << index << "(" << (int)ip << ") -> " << index_border << std::endl;

      HydroState2d u;
      if (is_ghost[iNeigh]) {
	u[ID] = U_ghost(index_border, fm[ID], neigh[iNeigh]);
	u[IP] = U_ghost(index_border, fm[IP], neigh[iNeigh]);
	u[IU] = U_ghost(index_border, fm[IU], neigh[iNeigh]);
	u[IV] = U_ghost(index_border, fm[IV], neigh[iNeigh]);
      } else {
	u[ID] += U(index_border, fm[ID], neigh[iNeigh]);
	u[IP] += U(index_border, fm[IP], neigh[iNeigh]);
	u[IU] += U(index_border, fm[IU], neigh[iNeigh]);
	u[IV] += U(index_border, fm[IV], neigh[iNeigh]);
      }

      // Converting to primitives
      real_t c = 0.0;
      computePrimitives(u, &c, q, params);
    } // end for ip
  }

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
   * Please note that indexes ic, ip and im refer to cell index
   * computed in the full ghosted block.
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

    const real_t drx = dqX[ID] * 0.5;
    const real_t dpx = dqX[IP] * 0.5;
    const real_t dux = dqX[IU] * 0.5;
    const real_t dvx = dqX[IV] * 0.5;
    
    const real_t dry = dqY[ID] * 0.5;
    const real_t dpy = dqY[IP] * 0.5;
    const real_t duy = dqY[IU] * 0.5;
    const real_t dvy = dqY[IV] * 0.5;
        
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
                                    uint32_t     iOct_local,
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

    const real_t drx = SlopesX(index,fm[ID],iOct_local) * 0.5;
    const real_t dpx = SlopesX(index,fm[IP],iOct_local) * 0.5;
    const real_t dux = SlopesX(index,fm[IU],iOct_local) * 0.5;
    const real_t dvx = SlopesX(index,fm[IV],iOct_local) * 0.5;
    //const real_t dwx = 0.0;
    
    const real_t dry = SlopesY(index,fm[ID],iOct_local) * 0.5;
    const real_t dpy = SlopesY(index,fm[IP],iOct_local) * 0.5;
    const real_t duy = SlopesY(index,fm[IU],iOct_local) * 0.5;
    const real_t dvy = SlopesY(index,fm[IV],iOct_local) * 0.5;
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
  void compute_slopes_2d(thread1_t member) const 
  {

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;
    
    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();
    
    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    while (iOct < iOctNextGroup and iOct < nbOcts)
    {

      /*
       * compute limited slopes
       */
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock1),
        KOKKOS_LAMBDA(const int32_t index) {

          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j
          const int j = index / bx1;
          const int i = index - j*bx1;

          // corresponding index in the full ghosted block
          // i -> i+1
          // j -> j+1
          const uint32_t ib = (i+1) + bx_g * (j+1);
          
          // neighbor along x axis
          uint32_t ibp1 = ib + 1;
          uint32_t ibm1 = ib - 1;
          
          SlopesX(ib,fm[ID],iOct_local) = 
            slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);
          
          SlopesX(ib,fm[IP],iOct_local) =
            slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);
          
          SlopesX(ib,fm[IU],iOct_local) =
            slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);
          
          SlopesX(ib,fm[IV],iOct_local) =
            slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);

          // neighbor along y axis
          ibp1 = ib + bx_g;
          ibm1 = ib - bx_g;
          
          SlopesY(ib,fm[ID],iOct_local) = 
            slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);
          SlopesY(ib,fm[IP],iOct_local) = 
            slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);
          SlopesY(ib,fm[IU],iOct_local) = 
            slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);
          SlopesY(ib,fm[IV],iOct_local) = 
            slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);

          // DEBUG : write into Ugroup
          //Ugroup(ib,fm[ID],iOct_local) = slopesX(ib,fm[ID]);

        }); // end TeamVectorRange

      iOct       += nbTeams;
      iOct_local += nbTeams;

    } // end while iOct < nbOct

  } // compute_slopes_2d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_2d_non_conformal(thread2_t member) const 
  {
    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;
    
    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();
    
    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    const uint32_t& bx = blockSizes[IX];
    const uint32_t& by = blockSizes[IY];
    
    while (iOct < iOctNextGroup and iOct < nbOcts) {
      
      // compute dx / dy
      const real_t dx = (iOct < nbOcts) ? pmesh->getSize(iOct)/bx : 1.0;
      const real_t dy = (iOct < nbOcts) ? pmesh->getSize(iOct)/by : 1.0;
      
      const real_t dtdx = dt/dx;
      const real_t dtdy = dt/dy;
      const real_t hdtdx = 0.5*dtdx;
      const real_t hdtdy = 0.5*dtdy;

      // For non conformal update (taken from muscl/ComputeFluxesAndUpdateHydroFunctor.h)

      // Building a list of cells to update
      std::set<coord_t, CoordComparator> cells_to_update;

      // We add the relevant cells to the set, these are indexed based on the original block
      if (Interface_flags(iOct_local) & INTERFACE_XMIN_NC)
	for (uint32_t j=0; j < by; ++j)
	  cells_to_update.insert(coord_t{0, j, 0});
      if (Interface_flags(iOct_local) & INTERFACE_XMAX_NC)
	for (uint32_t j=0; j < by; ++j)
	  cells_to_update.insert(coord_t{bx-1, j, 0});
      
      if (Interface_flags(iOct_local) & INTERFACE_YMIN_NC)
	for (uint32_t i=0; i < bx; ++i)
	  cells_to_update.insert(coord_t{i, 0, 0});
      if (Interface_flags(iOct_local) & INTERFACE_YMAX_NC)
	for (uint32_t i=0; i < bx; ++i)
	  cells_to_update.insert(coord_t{i, by-1, 0});

      // We convert the active cells to an ordered data structure for the Kokkos threads
      const int32_t nb_active_cells = cells_to_update.size();
      std::vector<coord_t> active_cells(nb_active_cells);
      std::copy(cells_to_update.begin(), cells_to_update.end(), active_cells.begin());

      // DEBUG TEST: Forcing non conservative update
      /*uint16_t new_interface = 0;
      if (Interface_flags(iOct_local) & (INTERFACE_XMIN_NC))
	new_interface |= INTERFACE_XMIN_BIGGER;
      if (Interface_flags(iOct_local) & (INTERFACE_YMIN_NC))
	new_interface |= INTERFACE_YMIN_BIGGER;
      if (Interface_flags(iOct_local) & (INTERFACE_XMAX_NC))
	new_interface |= INTERFACE_XMAX_BIGGER;
      if (Interface_flags(iOct_local) & (INTERFACE_YMAX_NC))
      new_interface |= INTERFACE_YMAX_BIGGER;

      Interface_flags(iOct_local) = new_interface;*/
      
      Kokkos::parallel_for(
	Kokkos::TeamVectorRange(member, nb_active_cells),
        KOKKOS_LAMBDA(const int32_t index) {
          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j
          const uint32_t ii = active_cells[index][0];
	  const uint32_t jj = active_cells[index][1];

	  // Position in the original block
          // corresponding index in the full ghosted block
          const uint32_t ig = (ii+ghostWidth) + bx_g * (jj+ghostWidth);
	  
	  // get current location primitive variables state
	  HydroState2d qprim = get_prim_variables<HydroState2d>(ig, iOct_local);
	  
	  // fluxes will be accumulated in qcons
	  HydroState2d qcons = {0.0, 0.0, 0.0, 0.0}; //get_cons_variables<HydroState2d>(ig, iOct_local);
	  
	  //std::cout << "qcons (start) : " << qcons[ID] << " " << qcons[IP] << " " << qcons[IU] << " " << qcons[IV] << std::endl;
	  
	  // Neighbour states for non conformal update
	  HydroState2d qm, qp;
	  
	  /*
	   * compute from left face along x dir
	   * if we are smaller than the neighbor, we update as before
	   */
	  if (ii==0 and (Interface_flags(iOct_local) & INTERFACE_XMIN_BIGGER)) {
	    // step 1 : reconstruct state in the left neighbor
	    
	    //std::cout << "  . X left non conformal bigger" << std::endl;
	    
	    // get state in neighbor along X
	    HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig-1, iOct_local);
	    
	    offsets_t offsets = {1.0, 0.0, 0.0};
	    
	    // reconstruct state in left neighbor
	    HydroState2d qL = reconstruct_state_2d(qprim_n, ig-1, iOct_local, offsets, dtdx, dtdy);
	    
	    // step 2 : reconstruct state in current cell
	    offsets = {-1.0, 0.0, 0.0};
	    
	    HydroState2d qR = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);
	    
	    // step 3 : compute flux (Riemann solver)
	    HydroState2d flux = riemann_hydro(qL,qR,params);
	    
	    // step 4 : accumulate flux in current cell
	    qcons += flux*dtdx;
	  }
	  // If we are bigger than the neighbors we sum two fluxes coming from the small cells
	  else if (ii==0 and (Interface_flags(iOct_local) & INTERFACE_XMIN_SMALLER)) {
	    // step 1 : Get the states of both neighbour cell
	    //std::cout << "  . X left non conformal smaller" << std::endl;
	    get_non_conformal_neighbors_2d(iOct, ii, jj, DIR_X, FACE_LEFT, qm, qp);
	    
	    // step 2 : Reconstruct state in current cell
	    offsets_t offsets = {-1.0, 0.0, 0.0};
	    HydroState2d qR = qprim; //reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);
	    
	    // step 3 : Solver is called directly on average states, no reconstruction is done
	    //          Warning: Should something be done here for the difference in size ?
	    HydroState2d flux_m = riemann_hydro(qm, qR, params);
	    HydroState2d flux_p = riemann_hydro(qp, qR, params);
	    qcons += flux_m*hdtdx;
	    qcons += flux_p*hdtdx;
	  }
	  
	  /*
	   * compute flux from right face along x dir
	   */
	  if (ii==bx-1 and (Interface_flags(iOct_local) & INTERFACE_XMAX_BIGGER)) {
	    // step 1 : reconstruct state in the left neighbor
	    
	    // get state in neighbor along X
	    HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig+1, iOct_local);
	    
	    // 
	    offsets_t offsets = {-1.0, 0.0, 0.0};
	    
	    // reconstruct state in right neighbor
	    HydroState2d qR = reconstruct_state_2d(qprim_n, ig+1, iOct_local, offsets, dtdx, dtdy);
	    
	    // step 2 : reconstruct state in current cell
	    offsets = {1.0, 0.0, 0.0};
	    
	    HydroState2d qL = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);
	    
	    // step 3 : compute flux (Riemann solver)
	    HydroState2d flux = riemann_hydro(qL,qR,params);
	    
	    // step 4 : accumulate flux in current cell
	    qcons -= flux*dtdx;
	  }
	  else if (ii==bx-1 and (Interface_flags(iOct_local) & INTERFACE_XMAX_SMALLER)) {
	    //std::cout << "  . X right non conformal smaller" << std::endl;
	    // step 1 : Get the states of both neighbour cells
	    get_non_conformal_neighbors_2d(iOct, ii, jj, DIR_X, FACE_RIGHT, qm, qp);
	    
	    // step 2 : Reconstruct state in current cell
	    offsets_t offsets = {1.0, 0.0, 0.0};
	    HydroState2d qL = qprim; //reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);
	    
	    // step 3 : Solver is called directly on average states, no reconstruction is done
	    //          Warning: Should something be done here for the difference in size ?
	    HydroState2d flux_m = riemann_hydro(qL, qm, params);
	    HydroState2d flux_p = riemann_hydro(qL, qp, params);
	    qcons -= flux_m*hdtdx;
	    qcons -= flux_p*hdtdx;
	  }
	  
	  /*
	   * compute flux from left face along y dir
	   */
	  if (jj==0 and (Interface_flags(iOct_local) & INTERFACE_YMIN_BIGGER)) {
	    // step 1 : reconstruct state in the left neighbor
	    
	    //std::cout << "  . Y left non conformal bigger" << std::endl;
	    
	    // get state in neighbor along X
	    HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig-bx_g, iOct_local);
	    
	    // 
	    offsets_t offsets = {0.0, 1.0, 0.0};
	    
	    // reconstruct "left" state
	    HydroState2d qL = reconstruct_state_2d(qprim_n, ig-bx_g, iOct_local, offsets, dtdx, dtdy);
	    
	    // step 2 : reconstruct state in current cell
	    offsets = {0.0, -1.0, 0.0};

	    HydroState2d qR = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);
	    
	    // swap IU / IV
	    my_swap(qL[IU], qL[IV]);
	    my_swap(qR[IU], qR[IV]);
	    
	    // step 3 : compute flux (Riemann solver)
	    HydroState2d flux = riemann_hydro(qL,qR,params);
	    
	    my_swap(flux[IU], flux[IV]);

	    // step 4 : accumulate flux in current cell
	    qcons += flux*dtdy;
	  }
	  else if (jj==0 and (Interface_flags(iOct_local) & INTERFACE_YMIN_SMALLER)) {
	    //std::cout << "  . Y left non conformal smaller" << std::endl;

	    // step 1 : Get the states of both neighbour cells
	    get_non_conformal_neighbors_2d(iOct, ii, jj, DIR_Y, FACE_LEFT, qm, qp);

	    // step 2 : Reconstruct state in current cell
	    offsets_t offsets = {0.0, -1.0, 0.0};
	    HydroState2d qR = qprim; //reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);

	    // swap IU / IV
	    my_swap(qp[IU], qp[IV]);
	    my_swap(qm[IU], qm[IV]);
	    my_swap(qR[IU], qR[IV]);

	    // step 3 : Solver is called directly on average states, no reconstruction is done
	    //          Warning: Should something be done here for the difference in size ?
	    HydroState2d flux_m = riemann_hydro(qm, qR, params);
	    HydroState2d flux_p = riemann_hydro(qp, qR, params);

	    my_swap(flux_p[IU], flux_p[IV]);
	    my_swap(flux_m[IU], flux_m[IV]);

	    qcons += flux_m*hdtdy;
	    qcons += flux_p*hdtdy;
	  }
	  
	  /*
	   * compute flux from right face along y dir
	   */
	  if (jj==by-1 and (Interface_flags(iOct_local) & INTERFACE_YMAX_BIGGER)) {
	    // step 1 : reconstruct state in the left neighbor
	    //std::cout << "  . Y right non conformal bigger" << std::endl;
	    
	    // get state in neighbor along X
	    HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig+bx_g, iOct_local);
	    
	    // 
	    offsets_t offsets = {0.0, -1.0, 0.0};
	    
	    // reconstruct "left" state
	    HydroState2d qR = reconstruct_state_2d(qprim_n, ig+bx_g, iOct_local, offsets, dtdx, dtdy);
	    
	    // step 2 : reconstruct state in current cell
	    offsets = {0.0, 1.0, 0.0};
	    
	    HydroState2d qL = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);
	    
	    // swap IU / IV
	    my_swap(qL[IU], qL[IV]);
	    my_swap(qR[IU], qR[IV]);
	    
	    // step 3 : compute flux (Riemann solver)
	    HydroState2d flux = riemann_hydro(qL,qR,params);
	    
	    my_swap(flux[IU], flux[IV]);
	    
	    // step 4 : accumulate flux in current cell
	    qcons -= flux*dtdy;
	  }
	  else if (jj==by-1 and (Interface_flags(iOct_local) & INTERFACE_YMAX_SMALLER)) {
	    //std::cout << "  . Y right non conformal smaller" << std::endl;
	    // step 1 : Get the states of both neighbour cells
	    get_non_conformal_neighbors_2d(iOct, ii, jj, DIR_Y, FACE_RIGHT, qm, qp);
	    
	    // step 2 : Reconstruct state in current cell
	    offsets_t offsets = {0.0, 1.0, 0.0};
	    HydroState2d qL = qprim; //reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);
	    
	    // swap IU / IV
	    my_swap(qL[IU], qL[IV]);
	    my_swap(qm[IU], qm[IV]);
	    my_swap(qp[IU], qp[IV]);
	    
	    // step 3 : Solver is called directly on average states, no reconstruction is done
	    //          Warning: Should something be done here for the difference in size ?
	    HydroState2d flux_m = riemann_hydro(qL, qm, params);
	    HydroState2d flux_p = riemann_hydro(qL, qp, params);

	    // Swap IU/IV
	    my_swap(flux_m[IU], flux_m[IV]);
	    my_swap(flux_p[IU], flux_p[IV]);
	    
	    qcons -= flux_m*hdtdy;
	    qcons -= flux_p*hdtdy;
	  }
	  
	  //std::cout << "qcons (end) : " << qcons[ID] << " " << qcons[IP] << " " << qcons[IU] << " " << qcons[IV] << std::endl;
	  
	  // lastly update conservative variable in U2
	  uint32_t index_non_ghosted = ii + bx * jj;
	  //std::cout << "index non ghosted = " << index_non_ghosted << " " << i << " " << j << " "
	  //	    << bx << " " << ghostWidth << std::endl;
	  
	  U2(index_non_ghosted, fm[ID], iOct) += qcons[ID];
	  U2(index_non_ghosted, fm[IP], iOct) += qcons[IP];
	  U2(index_non_ghosted, fm[IU], iOct) += qcons[IU];
	  U2(index_non_ghosted, fm[IV], iOct) += qcons[IV];
	}); // end TeamVectorRange
      
      
      iOct       += nbTeams;
      iOct_local += nbTeams;
      
    }// end while iOct < nbOct
	
  } // compute_fluxes_and_update_2d_non_conformal
    
  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_2d_conformal(thread2_t member) const 
  {
    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;
    
    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();
    
    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    const uint32_t& bx = blockSizes[IX];
    const uint32_t& by = blockSizes[IY];

    while (iOct < iOctNextGroup and iOct < nbOcts)
    {

      // compute dx / dy
      const real_t dx = (iOct < nbOcts) ? pmesh->getSize(iOct)/bx : 1.0;
      const real_t dy = (iOct < nbOcts) ? pmesh->getSize(iOct)/by : 1.0;
      
      const real_t dtdx = dt/dx;
      const real_t dtdy = dt/dy;
      const uint32_t bxm1 = bx-1;
      const uint32_t bym1 = by-1;
      
      /*
       * reconstruct states on cells face and update
       */
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock1),
        KOKKOS_LAMBDA(const int32_t index) {
          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j
          const uint32_t j = index / bx1;
          const uint32_t i = index - j*bx1;

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
            if (i > ghostWidth-1 or !(Interface_flags(iOct_local) & INTERFACE_XMIN_NC))
	    {
              // step 1 : reconstruct state in the left neighbor

              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig-1, iOct_local);

              // 
              offsets_t offsets = {1.0, 0.0, 0.0};

              // reconstruct state in left neighbor
              HydroState2d qL = reconstruct_state_2d(
                qprim_n, ig-1, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {-1.0, 0.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(
                qprim, ig, iOct_local, offsets, dtdx, dtdy);
              
              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              // step 4 : accumulate flux in current cell
              qcons += flux*dtdx;
            }

            /*
             * compute flux from right face along x dir
             */
	    if (i < bx1-ghostWidth or !(Interface_flags(iOct_local) & INTERFACE_XMAX_NC))
	      {
              // step 1 : reconstruct state in the left neighbor

              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig+1, iOct_local);

              // 
              offsets_t offsets = {-1.0, 0.0, 0.0};

              // reconstruct state in right neighbor
              HydroState2d qR = reconstruct_state_2d(
                qprim_n, ig+1, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {1.0, 0.0, 0.0};

              HydroState2d qL = reconstruct_state_2d(
                qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              // step 4 : accumulate flux in current cell
              qcons -= flux*dtdx;
            }

            /*
             * compute flux from left face along y dir
             */
            if (j > ghostWidth-1 or !(Interface_flags(iOct_local) & INTERFACE_YMIN_NC))
	      {
              // step 1 : reconstruct state in the left neighbor
              
              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig-bx_g, iOct_local);
              
              // 
              offsets_t offsets = {0.0, 1.0, 0.0};
              
              // reconstruct "left" state
              HydroState2d qL = reconstruct_state_2d(
                qprim_n, ig-bx_g, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {0.0, -1.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(
                qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // swap IU / IV
              my_swap(qL[IU], qL[IV]);
              my_swap(qR[IU], qR[IV]);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              my_swap(flux[IU], flux[IV]);

              // step 4 : accumulate flux in current cell
              qcons += flux*dtdy;
            }

            /*
             * compute flux from right face along y dir
             */
	    if (j < by1-ghostWidth or !(Interface_flags(iOct_local) & INTERFACE_YMAX_NC))
	      {
              // step 1 : reconstruct state in the left neighbor
              
              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig+bx_g, iOct_local);
              
              // 
              offsets_t offsets = {0.0, -1.0, 0.0};
              
              // reconstruct "left" state
              HydroState2d qR = reconstruct_state_2d(
                  qprim_n, ig+bx_g, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {0.0, 1.0, 0.0};

              HydroState2d qL = reconstruct_state_2d(
                qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // swap IU / IV
              my_swap(qL[IU], qL[IV]);
              my_swap(qR[IU], qR[IV]);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              my_swap(flux[IU], flux[IV]);

              // step 4 : accumulate flux in current cell
              qcons -= flux*dtdy;

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

  } // compute_fluxes_and_update_2d_conformal

    // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_2d_non_conservative(thread2_t member) const 
  {
    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;
    
    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();
    
    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    const uint32_t& bx = blockSizes[IX];
    const uint32_t& by = blockSizes[IY];

    while (iOct < iOctNextGroup and iOct < nbOcts)
    {

      // compute dx / dy
      const real_t dx = (iOct < nbOcts) ? pmesh->getSize(iOct)/bx : 1.0;
      const real_t dy = (iOct < nbOcts) ? pmesh->getSize(iOct)/by : 1.0;
      
      const real_t dtdx = dt/dx;
      const real_t dtdy = dt/dy;
      
      /*
       * reconstruct states on cells face and update
       */
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock1),
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

              // reconstruct state in left neighbor
              HydroState2d qL = reconstruct_state_2d(
                qprim_n, ig-1, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {-1.0, 0.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(
                qprim, ig, iOct_local, offsets, dtdx, dtdy);
              
              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              // step 4 : accumulate flux in current cell
              qcons += flux*dtdx;
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

              // reconstruct state in right neighbor
              HydroState2d qR = reconstruct_state_2d(
                qprim_n, ig+1, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {1.0, 0.0, 0.0};

              HydroState2d qL = reconstruct_state_2d(
                qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              // step 4 : accumulate flux in current cell
              qcons -= flux*dtdx;
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
                qprim_n, ig-bx_g, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {0.0, -1.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(
                qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // swap IU / IV
              my_swap(qL[IU], qL[IV]);
              my_swap(qR[IU], qR[IV]);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              my_swap(flux[IU], flux[IV]);

              // step 4 : accumulate flux in current cell
              qcons += flux*dtdy;
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
              HydroState2d qR = reconstruct_state_2d(
                  qprim_n, ig+bx_g, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {0.0, 1.0, 0.0};

              HydroState2d qL = reconstruct_state_2d(
                qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // swap IU / IV
              my_swap(qL[IU], qL[IV]);
              my_swap(qR[IU], qR[IV]);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL,qR,params);

              my_swap(flux[IU], flux[IV]);

              // step 4 : accumulate flux in current cell
              qcons -= flux*dtdy;

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

  } // compute_fluxes_and_update_2d_non_conservative

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_2d(thread2_t member) const 
  {
    if (params.updateType == UPDATE_CONSERVATIVE_SUM) {
      compute_fluxes_and_update_2d_conformal(member);
      compute_fluxes_and_update_2d_non_conformal(member);
    }
    else {
      compute_fluxes_and_update_2d_non_conservative(member);
    }
  } // compute_fluxes_and_update_2d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_slopes_3d(thread1_t member) const 
  {

  } // compute_slopes_3d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_3d(thread2_t member) const 
  {

  } // compute_fluxes_and_update_3d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const Slopes&, thread1_t member) const
  {
    
    if (this->params.dimType == TWO_D)
      compute_slopes_2d(member);
    
    else if (this->params.dimType == THREE_D)
      compute_slopes_3d(member);
    
  } // operator () - slopes

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const Fluxes&, thread2_t member) const
  {
    
    if (this->params.dimType == TWO_D)
      compute_fluxes_and_update_2d(member);
    
    else if (this->params.dimType == THREE_D)
      compute_fluxes_and_update_3d(member);
    
  } // operator () - fluxes and update

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

  //! blockSizes with one ghost cell 
  int32_t bx1;
  int32_t by1;
  int32_t bz1;

  //! ghost width
  uint32_t ghostWidth;

  //! total number of octants in current MPI process
  uint32_t nbOcts;

  //! number of octant per group
  uint32_t nbOctsPerGroup;

  //! number of cells per block (used for slopes computations)
  uint32_t nbCellsPerBlock1;

  //! integer which identifies a group of octants
  uint32_t iGroup;

  //! user data for the ith group of octants
  DataArrayBlock Ugroup;

  //! user data for the entire mesh
  DataArrayBlock U;
  
  //! user data for the neighbours
  DataArrayBlock U_ghost;

  //! user data for the entire mesh at the end of time step
  DataArrayBlock U2;

  //! user data (primitive variables) for the ith group of octants
  DataArrayBlock Qgroup;

  //! flags at interface for 2:1 ratio
  FlagArrayBlock Interface_flags;

  //! time step
  real_t         dt;

  //! slopes along x for current group
  DataArrayBlock SlopesX;

  //! slopes along y for current group
  DataArrayBlock SlopesY;

  //! slopes along z for current group
  DataArrayBlock SlopesZ;

}; // MusclBlockGodunovUpdateFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_GODUNOV_UPDATE_HYDRO_FUNCTOR_H_
