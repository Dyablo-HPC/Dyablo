/**
 * \file MusclBlockGodunovUpdateFunctor.h
 * \author Pierre Kestener
 */
#ifndef MUSCL_BLOCK_GODUNOV_UPDATE_HYDRO_FUNCTOR_H_
#define MUSCL_BLOCK_GODUNOV_UPDATE_HYDRO_FUNCTOR_H_

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "states/State_hydro.h"
#include "amr/LightOctree.h"
#include "RiemannSolvers.h"

// utils hydro
#include "utils_hydro.h"

// utils block
#include "legacy/utils_block.h"


namespace dyablo
{

namespace {

KOKKOS_INLINE_FUNCTION
const id2index_t& fm_state_3D()
{
  constexpr static id2index_t fm_state_3D_ = FieldManager( {ID,IP,IU,IV,IW} ).get_id2index();
  return fm_state_3D_;
}

KOKKOS_INLINE_FUNCTION
const id2index_t& fm_state_2D()
{
  constexpr static id2index_t fm_state_2D_ = FieldManager( {ID,IP,IU,IV} ).get_id2index();
  return fm_state_2D_;
}

template<typename State>
KOKKOS_INLINE_FUNCTION
const id2index_t& fm_state_ND();


template<>
KOKKOS_INLINE_FUNCTION
const id2index_t& fm_state_ND<HydroState3d>()
{
  return fm_state_3D();
}

template<>
KOKKOS_INLINE_FUNCTION
const id2index_t& fm_state_ND<HydroState2d>()
{
  return fm_state_2D();
}


KOKKOS_INLINE_FUNCTION
HydroState3d riemann_hydro( const HydroState3d& qleft_,
                            const HydroState3d& qright_,
                            const RiemannParams &params) 
{
  const auto& fm_state = fm_state_3D();

  PrimHydroState qleft{qleft_[fm_state[ID]], qleft_[fm_state[IP]], qleft_[fm_state[IU]], qleft_[fm_state[IV]], qleft_[fm_state[IW]]};
  PrimHydroState qright{qright_[fm_state[ID]], qright_[fm_state[IP]], qright_[fm_state[IU]], qright_[fm_state[IV]], qright_[fm_state[IW]]};
  ConsHydroState flux;
  riemann_hllc(qleft,qright,flux,params);
  HydroState3d flux_;
  flux_[fm_state[ID]] = flux.rho;
  flux_[fm_state[IP]] = flux.e_tot;
  flux_[fm_state[IU]] = flux.rho_u;
  flux_[fm_state[IV]] = flux.rho_v;
  flux_[fm_state[IW]] = flux.rho_w;
  return flux_;
}

KOKKOS_INLINE_FUNCTION
HydroState2d riemann_hydro( const HydroState2d& qleft_,
		                const HydroState2d& qright_,
                            const RiemannParams &params) 
{
  const auto& fm_state = fm_state_2D();

  PrimHydroState qleft{qleft_[fm_state[ID]], qleft_[fm_state[IP]], qleft_[fm_state[IU]], qleft_[fm_state[IV]], 0.0};
  PrimHydroState qright{qright_[fm_state[ID]], qright_[fm_state[IP]], qright_[fm_state[IU]], qright_[fm_state[IV]], 0.0};
  ConsHydroState flux;
  riemann_hllc(qleft,qright,flux,params);
  HydroState2d flux_;
  flux_[fm_state[ID]] = flux.rho;
  flux_[fm_state[IP]] = flux.e_tot;
  flux_[fm_state[IU]] = flux.rho_u;
  flux_[fm_state[IV]] = flux.rho_v;
  return flux_;
}

KOKKOS_INLINE_FUNCTION
void computePrimitives(const HydroState2d &u,
                       real_t             *c,
                       HydroState2d       &q,
                       real_t              gamma0,
                       real_t              smallr,
                       real_t              smallp)
{ 
  const auto& fm_state = fm_state_2D();

  real_t d, p, ux, uy;
  
  d = fmax(u[fm_state[ID]], smallr);
  ux = u[fm_state[IU]] / d;
  uy = u[fm_state[IV]] / d;
  
  // kinetic energy
  real_t eken = HALF_F * (ux*ux + uy*uy);
  
  // internal energy
  real_t e = u[fm_state[IE]] / d - eken;
  
  // compute pressure and speed of sound
  p = fmax((gamma0 - 1.0) * d * e, d * smallp);
  *c = sqrt(gamma0 * (p) / d);
  
  q[fm_state[ID]] = d;
  q[fm_state[IP]] = p;
  q[fm_state[IU]] = ux;
  q[fm_state[IV]] = uy; 
}

KOKKOS_INLINE_FUNCTION
void computePrimitives(const HydroState3d &u,
                       real_t             *c,
                       HydroState3d       &q,
                       real_t              gamma0,
                       real_t              smallr,
                       real_t              smallp)
{ 
  const auto& fm_state = fm_state_3D();

  real_t d, p, ux, uy, uz;
  
  d = fmax(u[fm_state[ID]], smallr);
  ux = u[fm_state[IU]] / d;
  uy = u[fm_state[IV]] / d;
  uz = u[fm_state[IW]] / d;
  
  // kinetic energy
  real_t eken = HALF_F * (ux*ux + uy*uy + uz*uz);
  
  // internal energy
  real_t e = u[fm_state[IE]] / d - eken;
  
  // compute pressure and speed of sound
  p = fmax((gamma0 - 1.0) * d * e, d * smallp);
  *c = sqrt(gamma0 * (p) / d);
  
  q[fm_state[ID]] = d;
  q[fm_state[IP]] = p;
  q[fm_state[IU]] = ux;
  q[fm_state[IV]] = uy;
  q[fm_state[IW]] = uz;  
}


template <class T>
KOKKOS_INLINE_FUNCTION 
void my_swap(T& a, T& b) {
  T c{std::move(a)};
  a = std::move(b);
  b = std::move(c);
} // my_swap

} // namespace

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
class MusclBlockGodunovUpdateFunctor
{
public:
  //! Tag structure for the 2 stages
  struct Slopes
  {
  }; // stage 1
  struct Fluxes
  {
  }; // stage 2
private:
  using offsets_t = Kokkos::Array<real_t, 3>;
  uint32_t nbTeams; //!< number of thread teams

  using team_policy1_t = Kokkos::TeamPolicy<Slopes, Kokkos::IndexType<int32_t>>;
  using thread1_t = team_policy1_t::member_type;

  using team_policy2_t = Kokkos::TeamPolicy<Fluxes, Kokkos::IndexType<int32_t>>;
  using thread2_t = team_policy2_t::member_type;

  // scratch memory aliases
  //using shared_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
  //using shared_2d_t  = Kokkos::View<real_t**, shared_space, Kokkos::MemoryUnmanaged>;

public:
  using index_t = int32_t;

  enum UPDATE_TYPE : uint16_t {
    UPDATE_NON_CONSERVATIVE = 0,
    UPDATE_CONSERVATIVE_SUM = 1
  };

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  struct Params{
    Params( ConfigMap& configMap )
    : riemann_params(configMap),
      gravity_type( configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE) ),
      gx( 0 ),
      gy( 0 ),
      gz( 0 ),
      slope_type( configMap.getValue<real_t>("hydro","slope_type",1.0) ),
      xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ),
      xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
      ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ),
      ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
      zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ),
      zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
      updateType(UPDATE_NON_CONSERVATIVE)
    {
      if (gravity_type & GRAVITY_CONSTANT) {
        gx = configMap.getValue<real_t>("gravity", "gx",  0.0);
        gy = configMap.getValue<real_t>("gravity", "gy", -1.0);
        gz = configMap.getValue<real_t>("gravity", "gz",  0.0);
      } 

      std::string utype = configMap.getValue<std::string>("hydro", "updateType", "conservative_sum");
      if (utype == "conservative_sum")
        updateType = UPDATE_CONSERVATIVE_SUM;
      else
        updateType = UPDATE_NON_CONSERVATIVE;
    }

    RiemannParams riemann_params;
    GravityType gravity_type;
    real_t gx, gy, gz;
    int slope_type;
    real_t xmin, xmax;
    real_t ymin, ymax;
    real_t zmin, zmax;
    UPDATE_TYPE updateType;
  };

  /**
   * Perform time integration (MUSCL Godunov).
   *
   * \param[in]  lmesh AMR mesh structure
   * \param[in]  params
   * \param[in]  fm field map
   * \param[in]  Ugroup current time step data (conservative variables)
   * \param[out] U2 next time step data (conservative variables)
   * \param[in]  Qgroup primitive variables
   * \param[in]  time step (as cmputed by CFL condition)
   *
   */
  MusclBlockGodunovUpdateFunctor(LightOctree lmesh,
                                 Params params,
                                 id2index_t fm,
                                 blockSize_t blockSizes,
                                 uint32_t ghostWidth,
                                 uint32_t nbOcts,
                                 uint32_t nbOctsPerGroup,
                                 uint32_t iGroup,
                                 DataArrayBlock Ugroup,
                                 LegacyDataArray U,
                                 LegacyDataArray U2,
                                 DataArrayBlock Qgroup,
                                 InterfaceFlags interface_flags,
                                 real_t dt) : lmesh(lmesh),
                                              params(params),
                                              fm(fm),
                                              blockSizes(blockSizes),
                                              ghostWidth(ghostWidth),
                                              nbOcts(nbOcts),
                                              nbOctsPerGroup(nbOctsPerGroup),
                                              iGroup(iGroup),
                                              Ugroup(Ugroup),
                                              U(U),
                                              U2(U2),
                                              Qgroup(Qgroup),
                                              interface_flags(interface_flags),
                                              dt(dt)
  {
    ndim = lmesh.getNdim();

    bx_g = blockSizes[IX] + 2 * (ghostWidth);
    by_g = blockSizes[IY] + 2 * (ghostWidth);
    bz_g = blockSizes[IZ] + 2 * (ghostWidth);

    // here we remove 1 to ghostWidth, only the inner
    // part need to compute limited slopes
    bx1 = blockSizes[IX] + 2 * (ghostWidth - 1);
    by1 = blockSizes[IY] + 2 * (ghostWidth - 1);
    bz1 = blockSizes[IZ] + 2 * (ghostWidth - 1);

    nbCellsPerBlock1 = ndim == 2 ? bx1 * by1 : bx1 * by1 * bz1;

    const uint32_t &bx = blockSizes[IX];
    const uint32_t &by = blockSizes[IY];
    const uint32_t &bz = blockSizes[IZ];
    nbCellsPerBlock = ndim == 2 ? bx * by : bx * by * bz;

    // memory allocation for slopes arrays
    SlopesX = DataArrayBlock("SlopesX",
                             Qgroup.extent(0),
                             Qgroup.extent(1),
                             Qgroup.extent(2));
    SlopesY = DataArrayBlock("SlopesY",
                             Qgroup.extent(0),
                             Qgroup.extent(1),
                             Qgroup.extent(2));
    if (ndim == 3)
      SlopesZ = DataArrayBlock("SlopesZ",
                               Qgroup.extent(0),
                               Qgroup.extent(1),
                               Qgroup.extent(2));

    has_gravity = (params.gravity_type != GRAVITY_NONE);

  }; // constructor

  // static method which does it all: create and execute functor
  static void apply(LightOctree lmesh,
                    Params params,
                    id2index_t fm,
                    blockSize_t blockSizes,
                    uint32_t ghostWidth,
                    uint32_t nbOcts,
                    uint32_t nbOctsPerGroup,
                    uint32_t iGroup,
                    DataArrayBlock Ugroup,
                    LegacyDataArray U,
                    LegacyDataArray U2,
                    DataArrayBlock Qgroup,
                    InterfaceFlags interface_flags,
                    real_t dt)
  {

    // instantiate functor
    MusclBlockGodunovUpdateFunctor functor(lmesh, params, fm,
                                           blockSizes, ghostWidth,
                                           nbOcts, nbOctsPerGroup, iGroup,
                                           Ugroup, U, U2,
                                           Qgroup,
                                           interface_flags,
                                           dt);

    uint32_t nbOctsInGroup = std::min( nbOctsPerGroup, lmesh.getNumOctants() - iGroup*nbOctsPerGroup );
    // create kokkos execution policy for slopes computation
    team_policy1_t policy1(nbOctsInGroup,
                           Kokkos::AUTO() /* team size */);

    // launch computation
    Kokkos::parallel_for("dyablo::muscl::MusclBlockGodunovUpdateFunctor - Slopes",
                         policy1, functor);

    // create kokkos execution policy for fluxes/update computation
    team_policy2_t policy2(nbOctsInGroup,
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
   * \param[out] q0 first neighboring state to be retrieved
   * \param[out] q1 second neighboring state
   */
  KOKKOS_INLINE_FUNCTION
  void get_non_conformal_neighbors_2d(uint32_t iOct,
                                      uint32_t i,
                                      uint32_t j,
                                      DIR_ID dir,
                                      FACE_ID face,
                                      HydroState2d &q0,
                                      HydroState2d &q1,
                                      GravityField &g0,
                                      GravityField &g1) const
  {
    const auto& fm_state = fm_state_2D();

    const uint32_t &bx = blockSizes[IX];
    const uint32_t &by = blockSizes[IY];

    LightOctree::offset_t offset = {0,0,0};
    offset[dir] = (face==FACE_LEFT) ? -1 : 1;

    // If non-conformal, there's necessarily a neighbor !
    LightOctree::NeighborList neighbors = lmesh.findNeighbors({iOct,false}, offset);

    uint32_t ii, jj; // Coords of the first neighbour
    uint8_t iNeigh = 0;

    if (dir == DIR_X)
    {
      ii = bx - i - 1;
      jj = j * 2;
    }
    else
    {
      jj = by - j - 1;
      ii = i * 2;
    }

    // We go through the two sub-cells
    for (uint8_t ip = 0; ip < 2; ++ip)
    {
      if (dir == DIR_X)
      {
        jj += ip;
        if (jj >= by)
        {
          iNeigh = 1;
          jj -= by;
        }
      }
      else
      {
        ii += ip;
        if (ii >= bx)
        {
          iNeigh = 1;
          ii -= bx;
        }
      }

      uint32_t index_border = ii + bx * jj;
      HydroState2d &q = (ip == 0 ? q0 : q1);
      GravityField &g = (ip == 0 ? g0 : g1);
      HydroState2d u;
      if (neighbors[iNeigh].isGhost)
      {
        u[fm_state[ID]] = U.ghost_val(index_border, fm[ID], neighbors[iNeigh].iOct);
        u[fm_state[IP]] = U.ghost_val(index_border, fm[IP], neighbors[iNeigh].iOct);
        u[fm_state[IU]] = U.ghost_val(index_border, fm[IU], neighbors[iNeigh].iOct);
        u[fm_state[IV]] = U.ghost_val(index_border, fm[IV], neighbors[iNeigh].iOct);

        if (params.gravity_type & GRAVITY_FIELD) {
          g[IX] = U.ghost_val(index_border, fm[IGX], neighbors[iNeigh].iOct);
          g[IY] = U.ghost_val(index_border, fm[IGY], neighbors[iNeigh].iOct);
        }
      }
      else
      {
        u[fm_state[ID]] = U(index_border, fm[ID], neighbors[iNeigh].iOct);
        u[fm_state[IP]] = U(index_border, fm[IP], neighbors[iNeigh].iOct);
        u[fm_state[IU]] = U(index_border, fm[IU], neighbors[iNeigh].iOct);
        u[fm_state[IV]] = U(index_border, fm[IV], neighbors[iNeigh].iOct);

        if (params.gravity_type & GRAVITY_FIELD) {
          g[IX] = U(index_border, fm[IGX], neighbors[iNeigh].iOct);
          g[IY] = U(index_border, fm[IGY], neighbors[iNeigh].iOct);
        }
      }

      if (params.gravity_type == GRAVITY_CST_SCALAR) {
        g[IX] = params.gx;
        g[IY] = params.gy;
      }

      // Converting to primitives
      real_t c = 0.0;
      computePrimitives(u, &c, q, 
        params.riemann_params.gamma0,
        params.riemann_params.smallr,
        params.riemann_params.smallp);
    } // end for ip

  } // get_non_conformal_neighbors_2d

  // ====================================================================
  // ====================================================================
  /**
   * Gets the gravity field inside a cell of the block
   * 
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant
   * 
   */
  KOKKOS_INLINE_FUNCTION
  GravityField 
  get_gravity_field(uint32_t index,
                    uint32_t iOct_local) const
  {
    GravityField res{0.0, 0.0, 0.0};
    if (params.gravity_type == GRAVITY_CST_SCALAR) {
      res[IX] = params.gx;
      res[IY] = params.gy;
      if (ndim == 3)
        res[IZ] = params.gz;
    }
    else if (params.gravity_type == GRAVITY_CST_FIELD) {
      res[IX] = Ugroup(index, fm[IGX], iOct_local);
      res[IY] = Ugroup(index, fm[IGY], iOct_local);
      if (ndim == 3)
        res[IZ] = Ugroup(index, fm[IGZ], iOct_local);
    }

    return res;

  } // get_gravity_field

  // ====================================================================
  // ====================================================================
  /**
   * Get conservative variables state vector.
   *
   * \param[in] index identifies location in the ghosted block
   * \param[in] iOct_local identifies octant (local index relative to
   *            a group of octant)
   */
  template <class HydroState>
  KOKKOS_INLINE_FUNCTION
  HydroState
  get_cons_variables(uint32_t index,
                     uint32_t iOct_local) const
  {
    const auto& fm_state = fm_state_ND<HydroState>();

    HydroState q;

    q[fm_state[ID]] = Ugroup(index, fm[ID], iOct_local);
    q[fm_state[IP]] = Ugroup(index, fm[IP], iOct_local);
    q[fm_state[IU]] = Ugroup(index, fm[IU], iOct_local);
    q[fm_state[IV]] = Ugroup(index, fm[IV], iOct_local);
    if (std::is_same<HydroState, HydroState3d>::value)
      q[fm_state[IW]] = Ugroup(index, fm[IW], iOct_local);

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
  template <class HydroState>
  KOKKOS_INLINE_FUNCTION
      HydroState
      get_prim_variables(uint32_t index,
                         uint32_t iOct_local) const
  {
    const auto& fm_state = fm_state_ND<HydroState>();

    HydroState q;

    q[fm_state[ID]] = Qgroup(index, fm[ID], iOct_local);
    q[fm_state[IP]] = Qgroup(index, fm[IP], iOct_local);
    q[fm_state[IU]] = Qgroup(index, fm[IU], iOct_local);
    q[fm_state[IV]] = Qgroup(index, fm[IV], iOct_local);
    if (std::is_same<HydroState, HydroState3d>::value)
      q[fm_state[IW]] = Qgroup(index, fm[IW], iOct_local);

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
    const real_t slope_type = params.slope_type;

    real_t dq = 0;

    if (slope_type == 1 or slope_type == 2)
    {

      // slopes in first coordinate direction
      const real_t dlft = slope_type * (q - qMinus);
      const real_t drgt = slope_type * (qPlus - q);
      const real_t dcen = HALF_F * (qPlus - qMinus);
      const real_t dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      if( std::isnan(dlft) or std::isnan(drgt)  ) return dlft+drgt;
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
    const real_t slope_type = params.slope_type;

    real_t dq = 0;

    if (slope_type == 1 or slope_type == 2)
    {

      const real_t q = Qgroup(ic, ivar, iOct_local);
      const real_t qPlus = Qgroup(ip, ivar, iOct_local);
      const real_t qMinus = Qgroup(im, ivar, iOct_local);

      // slopes in first coordinate direction
      const real_t dlft = slope_type * (q - qMinus);
      const real_t drgt = slope_type * (qPlus - q);
      const real_t dcen = HALF_F * (qPlus - qMinus);
      const real_t dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
      if( std::isnan(dlft) or std::isnan(drgt)  ) return dlft+drgt;
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
  template <class HydroState>
  KOKKOS_INLINE_FUNCTION
  HydroState
  slope_unsplit_hydro(const HydroState &q,
                      const HydroState &qPlus,
                      const HydroState &qMinus) const
  {

    //const real_t slope_type = params.slope_type;

    HydroState dq;

    const auto& fm_state = fm_state_ND<HydroState>();

    dq[fm_state[ID]] = slope_unsplit_scalar(q[fm_state[ID]], qPlus[fm_state[ID]], qMinus[fm_state[ID]]);
    dq[fm_state[IP]] = slope_unsplit_scalar(q[fm_state[IP]], qPlus[fm_state[IP]], qMinus[fm_state[IP]]);
    dq[fm_state[IU]] = slope_unsplit_scalar(q[fm_state[IU]], qPlus[fm_state[IU]], qMinus[fm_state[IU]]);
    dq[fm_state[IV]] = slope_unsplit_scalar(q[fm_state[IV]], qPlus[fm_state[IV]], qMinus[fm_state[IV]]);
    if (std::is_same<HydroState, HydroState3d>::value)
      dq[fm_state[IW]] = slope_unsplit_scalar(q[fm_state[IW]], qPlus[fm_state[IW]], qMinus[fm_state[IW]]);

    return dq;

  } // slope_unsplit_hydro

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void apply_gravity_prediction(HydroState2d &q, const GravityField &g) const
  {
    const auto& fm_state = fm_state_2D();

    q[fm_state[IU]] += 0.5 * dt * g[IX];
    q[fm_state[IV]] += 0.5 * dt * g[IY];
    //if (ndim == 3)
    //  q[fm_state[IW]] += 0.5 * dt * g[IZ];

  } // apply_gravity_prediction

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void apply_gravity_correction(uint32_t iOct_g, uint32_t index_g,
                                uint32_t iOct, uint32_t index) const
  {
    if (params.gravity_type == GRAVITY_NONE)
      return;

    real_t rhoOld = Ugroup(index_g, fm[ID], iOct_g);
    real_t rhoNew = fmax(params.riemann_params.smallr, U2(index, fm[ID], iOct));

    real_t rhou = U2(index, fm[IU], iOct);
    real_t rhov = U2(index, fm[IV], iOct);
    real_t rhow = (ndim == 3 ? U2(index, fm[IW], iOct) : 0.0);

    real_t ekin_old = 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rhoNew;

    real_t gx, gy, gz;
    if (params.gravity_type & GRAVITY_FIELD)
    {
      gx = Ugroup(index, fm[IGX], iOct_g);
      gy = Ugroup(index, fm[IGY], iOct_g);

      if (ndim == 3)
        gz = Ugroup(index_g, fm[IGZ], iOct_g);
    }
    else {
      gx = params.gx;
      gy = params.gy;

      if (ndim == 3)
        gz = params.gz;
    }

    rhou += 0.5 * dt * gx * (rhoOld + rhoNew);
    rhov += 0.5 * dt * gy * (rhoOld + rhoNew);
    U2(index, fm[IU], iOct) = rhou;
    U2(index, fm[IV], iOct) = rhov;

    if (ndim == 3)
    {
      rhow += 0.5 * dt * gz * (rhoOld + rhoNew);
      U2(index, fm[IW], iOct) = rhow;
    }

    real_t ekin_new = 0.5 * (rhou * rhou + rhov * rhov + rhow * rhow) / rhoNew;
    U2(index, fm[IP], iOct) += (ekin_new - ekin_old);

  }// apply_gravity_correction

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void apply_gravity_correction_2d(thread2_t member) const
  {
    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    const uint32_t &bx = blockSizes[IX];


    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock),
        [&](const int32_t index) {
          const uint32_t j = index / bx;
          const uint32_t i = index - j * bx;
          const uint32_t ig = (i + ghostWidth) + bx_g * (j + ghostWidth);

          apply_gravity_correction(iOct_local, ig, iOct, index);
        });
 

  } // apply_gravity_correction_2d

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
  HydroState2d reconstruct_state_2d(const HydroState2d &q,
                                    const HydroState2d &dqX,
                                    const HydroState2d &dqY,
                                    offsets_t offsets,
                                    real_t dtdx,
                                    real_t dtdy) const
  {
    const double gamma = params.riemann_params.gamma0;
    const double smallr = params.riemann_params.smallr;


    const auto& fm_state = fm_state_2D();

    // retrieve primitive variables in current quadrant
    const real_t r = q[fm_state[ID]];
    const real_t p = q[fm_state[IP]];
    const real_t u = q[fm_state[IU]];
    const real_t v = q[fm_state[IV]];

    const real_t drx = dqX[fm_state[ID]] * 0.5;
    const real_t dpx = dqX[fm_state[IP]] * 0.5;
    const real_t dux = dqX[fm_state[IU]] * 0.5;
    const real_t dvx = dqX[fm_state[IV]] * 0.5;

    const real_t dry = dqY[fm_state[ID]] * 0.5;
    const real_t dpy = dqY[fm_state[IP]] * 0.5;
    const real_t duy = dqY[fm_state[IU]] * 0.5;
    const real_t dvy = dqY[fm_state[IV]] * 0.5;

    // source terms (with transverse derivatives)
    const real_t sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
    const real_t su0 = (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
    const real_t sv0 = (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
    const real_t sp0 = (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;

    // reconstruct state on interface
    HydroState2d qr;

    qr[fm_state[ID]] = r + sr0 + offsets[IX] * drx + offsets[IY] * dry;
    qr[fm_state[IP]] = p + sp0 + offsets[IX] * dpx + offsets[IY] * dpy;
    qr[fm_state[IU]] = u + su0 + offsets[IX] * dux + offsets[IY] * duy;
    qr[fm_state[IV]] = v + sv0 + offsets[IX] * dvx + offsets[IY] * dvy;
    qr[fm_state[ID]] = fmax(smallr, qr[fm_state[ID]]);


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
  HydroState2d reconstruct_state_2d(const HydroState2d &q,
                                    uint32_t index,
                                    uint32_t iOct_local,
                                    offsets_t offsets,
                                    real_t dtdx,
                                    real_t dtdy) const
  {
    const double gamma = params.riemann_params.gamma0;
    const double smallr = params.riemann_params.smallr;

    const auto& fm_state = fm_state_2D();

    // retrieve primitive variables in current quadrant
    const real_t r = q[fm_state[ID]];
    const real_t p = q[fm_state[IP]];
    const real_t u = q[fm_state[IU]];
    const real_t v = q[fm_state[IV]];
    //const real_t w = 0.0;

    const real_t drx = SlopesX(index, fm[ID], iOct_local) * 0.5;
    const real_t dpx = SlopesX(index, fm[IP], iOct_local) * 0.5;
    const real_t dux = SlopesX(index, fm[IU], iOct_local) * 0.5;
    const real_t dvx = SlopesX(index, fm[IV], iOct_local) * 0.5;
    //const real_t dwx = 0.0;

    const real_t dry = SlopesY(index, fm[ID], iOct_local) * 0.5;
    const real_t dpy = SlopesY(index, fm[IP], iOct_local) * 0.5;
    const real_t duy = SlopesY(index, fm[IU], iOct_local) * 0.5;
    const real_t dvy = SlopesY(index, fm[IV], iOct_local) * 0.5;
    //const real_t dwy = 0.0;

    // source terms (with transverse derivatives)
    const real_t sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
    const real_t su0 = (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
    const real_t sv0 = (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
    const real_t sp0 = (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;

    // reconstruct state on interface
    HydroState2d qr;

    qr[fm_state[ID]] = r + sr0 + offsets[IX] * drx + offsets[IY] * dry;
    qr[fm_state[IP]] = p + sp0 + offsets[IX] * dpx + offsets[IY] * dpy;
    qr[fm_state[IU]] = u + su0 + offsets[IX] * dux + offsets[IY] * duy;
    qr[fm_state[IV]] = v + sv0 + offsets[IX] * dvx + offsets[IY] * dvy;
    qr[fm_state[ID]] = fmax(smallr, qr[fm_state[ID]]);

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
  HydroState3d reconstruct_state_3d(const HydroState3d& q,
                                    uint32_t index,
                                    uint32_t iOct_local,
                                    offsets_t offsets,
                                    real_t dtdx,
                                    real_t dtdy,
                                    real_t dtdz) const
  {
    const double gamma = params.riemann_params.gamma0;
    const double smallr = params.riemann_params.smallr;

    const auto& fm_state = fm_state_3D();

    // retrieve primitive variables in current quadrant
    const real_t r = q[fm_state[ID]];
    const real_t p = q[fm_state[IP]];
    const real_t u = q[fm_state[IU]];
    const real_t v = q[fm_state[IV]];
    const real_t w = q[fm_state[IW]];

    // retrieve variations = dx * slopes
    const real_t drx = SlopesX(index, fm[ID], iOct_local) * 0.5;
    const real_t dpx = SlopesX(index, fm[IP], iOct_local) * 0.5;
    const real_t dux = SlopesX(index, fm[IU], iOct_local) * 0.5;
    const real_t dvx = SlopesX(index, fm[IV], iOct_local) * 0.5;
    const real_t dwx = SlopesX(index, fm[IW], iOct_local) * 0.5;

    const real_t dry = SlopesY(index, fm[ID], iOct_local) * 0.5;
    const real_t dpy = SlopesY(index, fm[IP], iOct_local) * 0.5;
    const real_t duy = SlopesY(index, fm[IU], iOct_local) * 0.5;
    const real_t dvy = SlopesY(index, fm[IV], iOct_local) * 0.5;
    const real_t dwy = SlopesY(index, fm[IW], iOct_local) * 0.5;

    const real_t drz = SlopesZ(index, fm[ID], iOct_local) * 0.5;
    const real_t dpz = SlopesZ(index, fm[IP], iOct_local) * 0.5;
    const real_t duz = SlopesZ(index, fm[IU], iOct_local) * 0.5;
    const real_t dvz = SlopesZ(index, fm[IV], iOct_local) * 0.5;
    const real_t dwz = SlopesZ(index, fm[IW], iOct_local) * 0.5;

    // source terms (with transverse derivatives)
    const real_t sr0 = (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy + (-w * drz - dwz * r) * dtdz;
    const real_t su0 = (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy + (-w * duz) * dtdz;
    const real_t sv0 = (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy + (-w * dvz) * dtdz;
    const real_t sw0 = (-u * dwx) * dtdx + (-v * dwy) * dtdy + (-w * dwz - dpz / r) * dtdz;
    const real_t sp0 = (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy + (-w * dpz - dwz * gamma * p) * dtdz;

    // reconstruct state on interface
    HydroState3d qr;

    qr[fm_state[ID]] = r + sr0 + offsets[IX] * drx + offsets[IY] * dry + offsets[IZ] * drz;
    qr[fm_state[IP]] = p + sp0 + offsets[IX] * dpx + offsets[IY] * dpy + offsets[IZ] * dpz;
    qr[fm_state[IU]] = u + su0 + offsets[IX] * dux + offsets[IY] * duy + offsets[IZ] * duz;
    qr[fm_state[IV]] = v + sv0 + offsets[IX] * dvx + offsets[IY] * dvy + offsets[IZ] * dvz;
    qr[fm_state[IW]] = w + sw0 + offsets[IX] * dwx + offsets[IY] * dwy + offsets[IZ] * dwz;

    qr[fm_state[ID]] = fmax(smallr, qr[fm_state[ID]]);

    return qr;

  } // reconstruct_state_3d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_slopes_2d(thread1_t member) const
  {

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    // uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    /*
      * compute limited slopes
      */
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock1),
        [&](const int32_t index) {
          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j
          const int j = index / bx1;
          const int i = index - j * bx1;

          // corresponding index in the full ghosted block
          // i -> i+1
          // j -> j+1
          const uint32_t ib = (i + 1) + bx_g * (j + 1);

          // neighbor along x axis
          uint32_t ibp1 = ib + 1;
          uint32_t ibm1 = ib - 1;

          SlopesX(ib, fm[ID], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);

          SlopesX(ib, fm[IP], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);

          SlopesX(ib, fm[IU], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);

          SlopesX(ib, fm[IV], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);

          // neighbor along y axis
          ibp1 = ib + bx_g;
          ibm1 = ib - bx_g;

          SlopesY(ib, fm[ID], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);
          SlopesY(ib, fm[IP], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);
          SlopesY(ib, fm[IU], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);
          SlopesY(ib, fm[IV], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);

          // DEBUG : write into Ugroup
          //Ugroup(ib,fm[ID],iOct_local) = slopesX(ib,fm[ID]);
        }); // end TeamVectorRange

  } // compute_slopes_2d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_2d_non_conformal(thread2_t member) const
  {
    const auto& fm_state = fm_state_2D();

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    const uint32_t &bx = blockSizes[IX];
    const uint32_t &by = blockSizes[IY];
    const real_t Lx = params.xmax - params.xmin;
    const real_t Ly = params.ymax - params.ymin;

    // compute dx / dy
    auto oct_size = lmesh.getSize({iOct,false});
    const real_t dx = oct_size[IX] * Lx / bx;
    const real_t dy = oct_size[IY] * Ly / by;

    // TODO : Factor the update of U2 in an inline function instea of repeating the same block
    // of code again and again

    // Conformal interface area
    const real_t dSx_c = dy; // Interface when computing x flux is of length dy
    const real_t dSy_c = dx;

    // Non conformal interface area: divided by the number of neighbours
    const real_t dSx_nc = dSx_c * 0.5;
    const real_t dSy_nc = dSy_c * 0.5;

    // Volume
    const real_t dV = dx * dy;

    // Scaling for the flux in conformal and non conformal cases
    const real_t scale_x_c = dt * dSx_c / dV;
    const real_t scale_y_c = dt * dSy_c / dV;
    const real_t scale_x_nc = dt * dSx_nc / dV;
    const real_t scale_y_nc = dt * dSy_nc / dV;

    // We update the cells at the LEFT border if they have non-conformal neighbours
    if (interface_flags.isFaceNonConformal(iOct_local, FACE_LEFT))
    {
      const uint32_t ii = 0;
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, by),
          [&](const int32_t jj) {
            // Position in the original block
            // corresponding index in the full ghosted block
            const uint32_t ig = (ii + ghostWidth) + bx_g * (jj + ghostWidth);
            GravityField g;

            // get current location primitive variables state
            HydroState2d qprim = get_prim_variables<HydroState2d>(ig, iOct_local);
            
            // applying gravity on the primitive variable
            if (has_gravity) {
              g = get_gravity_field(ig, iOct_local);
              apply_gravity_prediction(qprim, g);
            }

            // fluxes will be accumulated in qcons
            HydroState2d qcons = {0.0, 0.0, 0.0, 0.0};
            if (interface_flags.isFaceBigger(iOct_local, FACE_LEFT))
            {
              // step 1: compute flux (Riemann solver) with centered values
              HydroState2d qR = qprim;
              HydroState2d qL = get_prim_variables<HydroState2d>(ig - 1, iOct_local);
                if (has_gravity) {
                g = get_gravity_field(ig-1, iOct_local);
                apply_gravity_prediction(qL, g);
              }

              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              // step 2: accumulate flux in current cell
              qcons += flux * scale_x_c;
            }
            // If we are bigger than the neighbors we sum two fluxes coming from the small cells
            else if (interface_flags.isFaceSmaller(iOct_local, FACE_LEFT))
            {
              // step 1: get the states of both neighbour cell
              HydroState2d qR = qprim;
              HydroState2d q0, q1;
              GravityField g0, g1;
              get_non_conformal_neighbors_2d(iOct, ii, jj, DIR_X, FACE_LEFT, q0, q1, g0, g1);

              if (has_gravity) {
                apply_gravity_prediction(q0, g0);
                apply_gravity_prediction(q1, g1);
              }

              // step 2: solver is called directly on average states, no reconstruction is done
              HydroState2d flux_0 = riemann_hydro(q0, qR, params.riemann_params);
              HydroState2d flux_1 = riemann_hydro(q1, qR, params.riemann_params);

              qcons += flux_0 * scale_x_nc;
              qcons += flux_1 * scale_x_nc;
            }

            // finally, update conservative variable in U2
            uint32_t index_non_ghosted = ii + bx * jj;

            U2(index_non_ghosted, fm[ID], iOct) += qcons[fm_state[ID]];
            U2(index_non_ghosted, fm[IP], iOct) += qcons[fm_state[IP]];
            U2(index_non_ghosted, fm[IU], iOct) += qcons[fm_state[IU]];
            U2(index_non_ghosted, fm[IV], iOct) += qcons[fm_state[IV]];
          });
    }

    // We update the cells at the RIGHT border if they have non-conformal neighbours
    if (interface_flags.isFaceNonConformal(iOct_local, FACE_RIGHT))
    {
      const uint32_t ii = bx - 1;
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, by),
          [&](const int32_t jj) {
            // Position in the original block
            // corresponding index in the full ghosted block
            const uint32_t ig = (ii + ghostWidth) + bx_g * (jj + ghostWidth);

            // get current location primitive variables state
            HydroState2d qprim = get_prim_variables<HydroState2d>(ig, iOct_local);
            GravityField g;
            if (has_gravity) {
              // applying gravity on the primitive variable
              g = get_gravity_field(ig, iOct_local);
              apply_gravity_prediction(qprim, g);
            }

            // fluxes will be accumulated in qcons
            HydroState2d qcons = {0.0, 0.0, 0.0, 0.0};
            if (interface_flags.isFaceBigger(iOct_local, FACE_RIGHT))
            {
              // step 1: compute flux (Riemann solver) with centered values
              HydroState2d qL = qprim;
              HydroState2d qR = get_prim_variables<HydroState2d>(ig + 1, iOct_local);

              if (has_gravity) {
                g = get_gravity_field(ig+1, iOct_local);
                apply_gravity_prediction(qR, g);
              }

              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              // step 2: accumulate flux in current cell
              qcons -= flux * scale_x_c;
            }
            else if (interface_flags.isFaceSmaller(iOct_local, FACE_RIGHT))
            {
              // step 1: get the states of both neighbour cells
              HydroState2d qL = qprim;
              HydroState2d q0, q1;
              GravityField g0, g1;
              get_non_conformal_neighbors_2d(iOct, ii, jj, DIR_X, FACE_RIGHT, q0, q1, g0, g1);

              if (has_gravity) {
                apply_gravity_prediction(q0, g0);
                apply_gravity_prediction(q1, g1);
              }

              // step 2: solver is called directly on average states, no reconstruction is done
              HydroState2d flux_0 = riemann_hydro(qL, q0, params.riemann_params);
              HydroState2d flux_1 = riemann_hydro(qL, q1, params.riemann_params);

              // step 3: accumulate
              qcons -= flux_0 * scale_x_nc;
              qcons -= flux_1 * scale_x_nc;
            }

            // finally, update conservative variable in U2
            uint32_t index_non_ghosted = ii + bx * jj;

            U2(index_non_ghosted, fm[ID], iOct) += qcons[fm_state[ID]];
            U2(index_non_ghosted, fm[IP], iOct) += qcons[fm_state[IP]];
            U2(index_non_ghosted, fm[IU], iOct) += qcons[fm_state[IU]];
            U2(index_non_ghosted, fm[IV], iOct) += qcons[fm_state[IV]];
          });
    }

    // We update the cells at the BOTTOM border if they have non-conformal neighbours
    if (interface_flags.isFaceNonConformal(iOct_local, FACE_BOTTOM))
    {
      const uint32_t jj = 0;
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, bx),
          [&](const int32_t ii) {
            // Position in the original block
            // corresponding index in the full ghosted block
            const uint32_t ig = (ii + ghostWidth) + bx_g * (jj + ghostWidth);

            // get current location primitive variables state
            HydroState2d qprim = get_prim_variables<HydroState2d>(ig, iOct_local);
            GravityField g;

            if (has_gravity) {
              // applying gravity on the primitive variable
              g = get_gravity_field(ig, iOct_local);
              apply_gravity_prediction(qprim, g);
            }

            // fluxes will be accumulated in qcons
            HydroState2d qcons = {0.0, 0.0, 0.0, 0.0};

            if (interface_flags.isFaceBigger(iOct_local, FACE_BOTTOM))
            {
              // step 1: Swap u and v in states
              HydroState2d qL = get_prim_variables<HydroState2d>(ig - bx_g, iOct_local);
              HydroState2d qR = qprim;

              if (has_gravity) {
                g = get_gravity_field(ig-bx_g, iOct_local);
                apply_gravity_prediction(qL, g);
              }

              my_swap(qL[fm_state[IU]], qL[fm_state[IV]]);
              my_swap(qR[fm_state[IU]], qR[fm_state[IV]]);

              // step 2: compute flux (Riemann solver) with centered values
              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              // step 3: swap back and accumulate flux
              my_swap(flux[fm_state[IU]], flux[fm_state[IV]]);
              qcons += flux * scale_y_c;
            }
            else if (interface_flags.isFaceSmaller(iOct_local, FACE_BOTTOM))
            {
              // step 1: get the states of both neighbour cells
              HydroState2d qR = qprim;
              HydroState2d q0, q1;
              GravityField g0, g1;
              get_non_conformal_neighbors_2d(iOct, ii, jj, DIR_Y, FACE_LEFT, q0, q1, g0, g1);

              if (has_gravity) {
                apply_gravity_prediction(q0, g0);
                apply_gravity_prediction(q1, g1);
              }

              // step 2: u and v in states
              my_swap(q0[fm_state[IU]], q0[fm_state[IV]]);
              my_swap(q1[fm_state[IU]], q1[fm_state[IV]]);
              my_swap(qR[fm_state[IU]], qR[fm_state[IV]]);

              // step 3: solver is called directly on average states, no reconstruction is done
              HydroState2d flux_0 = riemann_hydro(q0, qR, params.riemann_params);
              HydroState2d flux_1 = riemann_hydro(q1, qR, params.riemann_params);

              // step 4: swap back and accumulate
              my_swap(flux_0[fm_state[IU]], flux_0[fm_state[IV]]);
              my_swap(flux_1[fm_state[IU]], flux_1[fm_state[IV]]);

              qcons += flux_0 * scale_y_nc;
              qcons += flux_1 * scale_y_nc;
            }

            // finally, update conservative variable in U2
            uint32_t index_non_ghosted = ii + bx * jj;

            U2(index_non_ghosted, fm[ID], iOct) += qcons[fm_state[ID]];
            U2(index_non_ghosted, fm[IP], iOct) += qcons[fm_state[IP]];
            U2(index_non_ghosted, fm[IU], iOct) += qcons[fm_state[IU]];
            U2(index_non_ghosted, fm[IV], iOct) += qcons[fm_state[IV]];
          });
    }

    // We update the cells at the TOP border if they have non-conformal neighbours
    if (interface_flags.isFaceNonConformal(iOct_local, FACE_TOP))
    {
      const uint32_t jj = by - 1;
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, bx),
          [&](const int32_t ii) {
            // Position in the original block
            // corresponding index in the full ghosted block
            const uint32_t ig = (ii + ghostWidth) + bx_g * (jj + ghostWidth);

            // get current location primitive variables state
            HydroState2d qprim = get_prim_variables<HydroState2d>(ig, iOct_local);
            GravityField g;

            if (has_gravity) {
              // applying gravity on the primitive variable
              g = get_gravity_field(ig, iOct_local);
              apply_gravity_prediction(qprim, g);
            }
            // fluxes will be accumulated in qcons
            HydroState2d qcons = {0.0, 0.0, 0.0, 0.0};

            if (interface_flags.isFaceBigger(iOct_local, FACE_TOP))
            {
              // step 1: Swap u and v in states
              HydroState2d qR = get_prim_variables<HydroState2d>(ig + bx_g, iOct_local);
              HydroState2d qL = qprim;

              if (has_gravity) {
                g = get_gravity_field(ig+bx_g, iOct_local);
                apply_gravity_prediction(qR, g);
              }
              my_swap(qL[fm_state[IU]], qL[fm_state[IV]]);
              my_swap(qR[fm_state[IU]], qR[fm_state[IV]]);

              // step 2: compute flux (Riemann solver) with centered values
              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              // step 3: swap back and accumulate flux
              my_swap(flux[fm_state[IU]], flux[fm_state[IV]]);
              qcons -= flux * scale_y_c;
            }
            else if (interface_flags.isFaceSmaller(iOct_local, FACE_TOP))
            {
              // step1: get the states of both neighbour cells
              HydroState2d qL = qprim;
              HydroState2d q0, q1;
              GravityField g0, g1;
              get_non_conformal_neighbors_2d(iOct, ii, jj, DIR_Y, FACE_RIGHT, q0, q1, g0, g1);

              if (has_gravity) {
                apply_gravity_prediction(q0, g0);
                apply_gravity_prediction(q1, g1);
              }

              // step 2: swap u and v in states
              my_swap(qL[fm_state[IU]], qL[fm_state[IV]]);
              my_swap(q0[fm_state[IU]], q0[fm_state[IV]]);
              my_swap(q1[fm_state[IU]], q1[fm_state[IV]]);

              // step 3 : solver is called directly on average states, no reconstruction is done
              HydroState2d flux_0 = riemann_hydro(qL, q0, params.riemann_params);
              HydroState2d flux_1 = riemann_hydro(qL, q1, params.riemann_params);

              // step 4: swap back and accumulate
              my_swap(flux_0[fm_state[IU]], flux_0[fm_state[IV]]);
              my_swap(flux_1[fm_state[IU]], flux_1[fm_state[IV]]);

              qcons -= flux_0 * scale_y_nc;
              qcons -= flux_1 * scale_y_nc;
            }

            // finally, update conservative variable in U2
            uint32_t index_non_ghosted = ii + bx * jj;

            U2(index_non_ghosted, fm[ID], iOct) += qcons[fm_state[ID]];
            U2(index_non_ghosted, fm[IP], iOct) += qcons[fm_state[IP]];
            U2(index_non_ghosted, fm[IU], iOct) += qcons[fm_state[IU]];
            U2(index_non_ghosted, fm[IV], iOct) += qcons[fm_state[IV]];
          });
    }

  } // compute_fluxes_and_update_2d_non_conformal

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_2d_conformal(thread2_t member) const
  {
    const auto& fm_state = fm_state_2D();

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    const uint32_t &bx = blockSizes[IX];
    const uint32_t &by = blockSizes[IY];
    const real_t Lx = params.xmax - params.xmin;
    const real_t Ly = params.ymax - params.ymin;


    // compute dx / dy
    auto oct_size = lmesh.getSize({iOct,false});
    const real_t dx = oct_size[IX] * Lx / bx;
    const real_t dy = oct_size[IY] * Ly / by;

    const real_t dtdx = dt / dx;
    const real_t dtdy = dt / dy;

    /*
      * reconstruct states on cells face and update
      */
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock),
        [&](const int32_t index) {
          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j
          const uint32_t j = index / bx;
          const uint32_t i = index - j * bx;

          // corresponding index in the full ghosted block
          // i -> i+1
          // j -> j+1
          const uint32_t ig = (i + ghostWidth) + bx_g * (j + ghostWidth);

          // the following condition makes sure we stay inside
          // the inner block
          if (/*i >= 0 and*/ i < bx and
              /*j >= 0 and*/ j < by)
          {
            // get current location primitive variables state
            HydroState2d qprim = get_prim_variables<HydroState2d>(ig, iOct_local);

            GravityField gc = get_gravity_field(ig, iOct_local);
            GravityField g;

            // fluxes will be accumulated in qcons
            HydroState2d qcons = get_cons_variables<HydroState2d>(ig, iOct_local);

            auto process_axis = [&]( ComponentIndex3D dir, FACE_ID face)
            {
              offsets_t offsets = {0.,0.,0.};
              int32_t face_sign = (face == FACE_LEFT) ? -1 : 1;
              offsets[dir] = -face_sign; 
              int32_t ioffset = - ( offsets[IX] + bx_g * offsets[IY] );

              // get state in neighbor along dir
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig + ioffset, iOct_local);

              // reconstruct "left/right" state
              HydroState2d qout =
                  reconstruct_state_2d(qprim_n, ig + ioffset, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell (offset is negated compared to other cell)
              offsets[IX] = - offsets[IX];
              offsets[IY] = - offsets[IY];
              offsets[IZ] = - offsets[IZ];
              HydroState2d qin = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // step 3 : compute gravity
              if (has_gravity) {
                apply_gravity_prediction(qin, gc); // Prediction for local cell
                g = get_gravity_field(ig + ioffset, iOct_local);
                apply_gravity_prediction(qout, g); // Prediction for neighbor cell
              }

              HydroState2d& qL = (face==FACE_LEFT) ? qout : qin;
              HydroState2d& qR = (face==FACE_LEFT) ? qin : qout;

              if( dir == IY )
            {
              // swap IU / IV
              my_swap(qL[fm_state[IU]], qL[fm_state[IV]]);
              my_swap(qR[fm_state[IU]], qR[fm_state[IV]]);
              }

              // step 4 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              if( dir == IY )
              my_swap(flux[fm_state[IU]], flux[fm_state[IV]]);

              // step 5 : accumulate flux in current cell
              qcons -= flux * ( offsets[IX] * dtdx + offsets[IY] * dtdy ) ;
            };

            // compute from left face along x dir
            if (i > 0 or interface_flags.isFaceConformal(iOct_local, FACE_LEFT))
            {
              process_axis(IX, FACE_LEFT);
            }
            // compute flux from right face along x dir
            if (i < bx - 1 or interface_flags.isFaceConformal(iOct_local, FACE_RIGHT))
            {
              process_axis(IX, FACE_RIGHT);
            }
            // compute flux from left face along y dir
            if (j > 0 or interface_flags.isFaceConformal(iOct_local, FACE_BOTTOM))
            {
              process_axis(IY, FACE_LEFT);
            }
            // compute flux from right face along y dir
            if (j < by - 1 or interface_flags.isFaceConformal(iOct_local, FACE_TOP))
            {
              process_axis(IY, FACE_RIGHT);
            }

            // lastly update conservative variable in U2
            uint32_t index_non_ghosted = i + bx * j; //(i-1) + bx * (j-1);

            U2(index_non_ghosted, fm[ID], iOct) = qcons[fm_state[ID]];
            U2(index_non_ghosted, fm[IP], iOct) = qcons[fm_state[IP]];
            U2(index_non_ghosted, fm[IU], iOct) = qcons[fm_state[IU]];
            U2(index_non_ghosted, fm[IV], iOct) = qcons[fm_state[IV]];

          } // end if inside inner block
        }); // end TeamVectorRange


  } // compute_fluxes_and_update_2d_conformal

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_2d_non_conservative(thread2_t member) const
  {
    const auto& fm_state = fm_state_2D();

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    const uint32_t &bx = blockSizes[IX];
    const uint32_t &by = blockSizes[IY];

    // compute dx / dy
    auto oct_size = lmesh.getSize({iOct,false});
    const real_t dx = (iOct < nbOcts) ? oct_size[IX] / bx : 1.0;
    const real_t dy = (iOct < nbOcts) ? oct_size[IY] / by : 1.0;

    const real_t dtdx = dt / dx;
    const real_t dtdy = dt / dy;

    /*
      * reconstruct states on cells face and update
      */
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock1),
        [&](const int32_t index) {
          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j
          const int j = index / bx1;
          const int i = index - j * bx1;

          // corresponding index in the full ghosted block
          // i -> i+1
          // j -> j+1
          const uint32_t ig = (i + 1) + bx_g * (j + 1);

          // the following condition makes sure we stay inside
          // the inner block
          if (i > 0 and i < bx1 - 1 and
              j > 0 and j < by1 - 1)
          {
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
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig - 1, iOct_local);

              //
              offsets_t offsets = {1.0, 0.0, 0.0};

              // reconstruct state in left neighbor
              HydroState2d qL = reconstruct_state_2d(qprim_n, ig - 1, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {-1.0, 0.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              // step 4 : accumulate flux in current cell
              qcons += flux * dtdx;
            }

            /*
              * compute flux from right face along x dir
              */
            {
              // step 1 : reconstruct state in the left neighbor

              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig + 1, iOct_local);

              //
              offsets_t offsets = {-1.0, 0.0, 0.0};

              // reconstruct state in right neighbor
              HydroState2d qR = reconstruct_state_2d(qprim_n, ig + 1, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {1.0, 0.0, 0.0};

              HydroState2d qL = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              // step 4 : accumulate flux in current cell
              qcons -= flux * dtdx;
            }

            /*
              * compute flux from left face along y dir
              */
            {
              // step 1 : reconstruct state in the left neighbor

              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig - bx_g, iOct_local);

              //
              offsets_t offsets = {0.0, 1.0, 0.0};

              // reconstruct "left" state
              HydroState2d qL = reconstruct_state_2d(qprim_n, ig - bx_g, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {0.0, -1.0, 0.0};

              HydroState2d qR = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // swap IU / IV
              my_swap(qL[fm_state[IU]], qL[fm_state[IV]]);
              my_swap(qR[fm_state[IU]], qR[fm_state[IV]]);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              my_swap(flux[fm_state[IU]], flux[fm_state[IV]]);

              // step 4 : accumulate flux in current cell
              qcons += flux * dtdy;
            }

            /*
              * compute flux from right face along y dir
              */
            {
              // step 1 : reconstruct state in the left neighbor

              // get state in neighbor along X
              HydroState2d qprim_n = get_prim_variables<HydroState2d>(ig + bx_g, iOct_local);

              //
              offsets_t offsets = {0.0, -1.0, 0.0};

              // reconstruct "left" state
              HydroState2d qR = reconstruct_state_2d(qprim_n, ig + bx_g, iOct_local, offsets, dtdx, dtdy);

              // step 2 : reconstruct state in current cell
              offsets = {0.0, 1.0, 0.0};

              HydroState2d qL = reconstruct_state_2d(qprim, ig, iOct_local, offsets, dtdx, dtdy);

              // swap IU / IV
              my_swap(qL[fm_state[IU]], qL[fm_state[IV]]);
              my_swap(qR[fm_state[IU]], qR[fm_state[IV]]);

              // step 3 : compute flux (Riemann solver)
              HydroState2d flux = riemann_hydro(qL, qR, params.riemann_params);

              my_swap(flux[fm_state[IU]], flux[fm_state[IV]]);

              // step 4 : accumulate flux in current cell
              qcons -= flux * dtdy;
            }

            // lastly update conservative variable in U2
            uint32_t index_non_ghosted = (i - 1) + bx * (j - 1);

            U2(index_non_ghosted, fm[ID], iOct) = qcons[fm_state[ID]];
            U2(index_non_ghosted, fm[IP], iOct) = qcons[fm_state[IP]];
            U2(index_non_ghosted, fm[IU], iOct) = qcons[fm_state[IU]];
            U2(index_non_ghosted, fm[IV], iOct) = qcons[fm_state[IV]];

          } // end if inside inner block
        }); // end TeamVectorRange

  } // compute_fluxes_and_update_2d_non_conservative

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_2d(thread2_t member) const
  {
    if (params.updateType == UPDATE_CONSERVATIVE_SUM)
    {
      compute_fluxes_and_update_2d_conformal(member);
      //compute_fluxes_and_update_2d_non_conformal(member);
    }
    else
    {
      compute_fluxes_and_update_2d_non_conservative(member);
    }

    if (has_gravity) {
      apply_gravity_correction_2d(member);
    }
  } // compute_fluxes_and_update_2d

  // ====================================================================
  // ====================================================================
  /**
   * Applies predictor step for gravity
   * @param index index of the cell in current block
   * @param iOct_local index of the octant in local group
   * @param dt time step
   **/
  // KOKKOS_INLINE_FUNCTION
  // void apply_gravity_prediction(HydroState3d &q, 
  //                               uint32_t index,
  //                               uint32_t iOct_local,  
  //                               real_t dt) const {
  //   q[fm_state[IU]] += 0.5 * dt * Ugroup(index, fm[IGX], iOct_local);
  //   q[fm_state[IV]] += 0.5 * dt * Ugroup(index, fm[IGY], iOct_local);
  //   q[fm_state[IW]] += 0.5 * dt * Ugroup(index, fm[IGZ], iOct_local);
  // }

  /**
   * Applies corrector step for gravity
   * @param index index of the cell in current block
   * @param iOct_local index of the octant in local group
   * @param dt time step
   **/
  KOKKOS_INLINE_FUNCTION
  void apply_gravity_correction_3d(HydroState3d &qOld,
                                HydroState3d &qNew, 
                                uint32_t index, 
                                uint32_t iOct_local, 
                                real_t dt) const {

    const auto& fm_state = fm_state_3D();

    const real_t gx = Ugroup(index, fm[IGX], iOct_local);
    const real_t gy = Ugroup(index, fm[IGY], iOct_local);
    const real_t gz = Ugroup(index, fm[IGZ], iOct_local);

    real_t rhoOld = qOld[fm_state[ID]];
    real_t rhoNew = qNew[fm_state[ID]];

    real_t rhou = qNew[fm_state[IU]];
    real_t rhov = qNew[fm_state[IV]];
    real_t rhow = qNew[fm_state[IW]];

    real_t ekin_old = 0.5 * (rhou*rhou + rhov*rhov + rhow*rhow) / rhoNew;

    rhou += 0.5 * dt * gx * (rhoOld + rhoNew);
    rhov += 0.5 * dt * gy * (rhoOld + rhoNew);

    qNew[fm_state[IU]] = rhou;
    qNew[fm_state[IV]] = rhov;
    //if (ndim == 3) {
      rhow += 0.5 * dt * gz * (rhoOld + rhoNew);
      qNew[fm_state[IW]] = rhow;
    //}

    // Energy correction should be included in case of self-gravitation ?
    real_t ekin_new = 0.5 * (rhou*rhou + rhov*rhov + rhow*rhow) / rhoNew;
    qNew[fm_state[IE]] += (ekin_new - ekin_old);
  }

  KOKKOS_INLINE_FUNCTION
  void compute_slopes_3d(thread1_t member) const
  {
    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();
    /*
      * compute limited slopes
      */
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock1),
        [&](const int32_t index) {
          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx1 * j + bx1 * by1 * k
          const int k = index / (bx1*by1);
          const int j = (index - k*bx1*by1 )/ bx1;
          const int i = index - j * bx1 - k*bx1*by1 ;

          // corresponding index in the full ghosted block
          // i -> i+1
          // j -> j+1
          const uint32_t ib = (i + 1) + bx_g * (j + 1) + by_g * bx_g  *( k + 1 );

          // neighbor along x axis
          uint32_t ibp1 = ib + 1;
          uint32_t ibm1 = ib - 1;

          SlopesX(ib, fm[ID], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);
          SlopesX(ib, fm[IP], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);
          SlopesX(ib, fm[IU], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);
          SlopesX(ib, fm[IV], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);
          SlopesX(ib, fm[IW], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IW], iOct_local);

          // neighbor along y axis
          ibp1 = ib + bx_g;
          ibm1 = ib - bx_g;

          SlopesY(ib, fm[ID], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);
          SlopesY(ib, fm[IP], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);
          SlopesY(ib, fm[IU], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);
          SlopesY(ib, fm[IV], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);
          SlopesY(ib, fm[IW], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IW], iOct_local);

          // neighbor along y axis
          ibp1 = ib + bx_g*by_g;
          ibm1 = ib - bx_g*by_g;

          SlopesZ(ib, fm[ID], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[ID], iOct_local);
          SlopesZ(ib, fm[IP], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IP], iOct_local);
          SlopesZ(ib, fm[IU], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IU], iOct_local);
          SlopesZ(ib, fm[IV], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IV], iOct_local);
          SlopesZ(ib, fm[IW], iOct_local) =
              slope_unsplit_scalar(ib, ibp1, ibm1, fm[IW], iOct_local);

          // DEBUG : write into Ugroup
          //Ugroup(ib,fm[ID],iOct_local) = slopesX(ib,fm[ID]);
        }); // end TeamVectorRange

  } // compute_slopes_3d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_3d_conformal(thread2_t member) const
  {
    const auto& fm_state = fm_state_3D();

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    const uint32_t &bx = blockSizes[IX];
    const uint32_t &by = blockSizes[IY];
    const uint32_t &bz = blockSizes[IZ];
    const real_t Lx = params.xmax - params.xmin;
    const real_t Ly = params.ymax - params.ymin;
    const real_t Lz = params.zmax - params.zmin;

    // compute dx / dy
    auto oct_size = lmesh.getSize({iOct,false});
    const real_t dx = oct_size[IX] * Lx / bx;
    const real_t dy = oct_size[IY] * Ly / by;
    const real_t dz = oct_size[IZ] * Lz / bz;

    const real_t dtdx = dt / dx;
    const real_t dtdy = dt / dy;
    const real_t dtdz = dt / dz;

    /*
      * reconstruct states on cells face and update
      */
    Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCellsPerBlock),
        [&](const int32_t index) {
          // convert index to coordinates in ghosted block (minus 1 !)
          //index = i + bx * j + bx * by * k
          const uint32_t k = index / (bx*by);
          const uint32_t j = (index - k*bx*by) / bx;
          const uint32_t i = index - k*bx*by - j*bx;

          // corresponding index in the full ghosted block
          // i -> i+1
          // j -> j+1
          const uint32_t ig = (i + ghostWidth) + bx_g * (j + ghostWidth) + bx_g * by_g * (k + ghostWidth);

          // the following condition makes sure we stay inside
          // the inner block
          if (/*i >= 0 and*/ i < bx and
              /*j >= 0 and*/ j < by and
              /*k >= 0 and*/ k < bz )
          {
            // get current location primitive variables state
            HydroState3d qprim = get_prim_variables<HydroState3d>(ig, iOct_local);

            // fluxes will be accumulated in qcons
            HydroState3d qcons = get_cons_variables<HydroState3d>(ig, iOct_local);
            HydroState3d qold  = qcons;

            auto process_axis = [&]( ComponentIndex3D dir, FACE_ID face)
            {
              offsets_t offsets = {0.,0.,0.};
              int32_t face_sign = (face == FACE_LEFT) ? -1 : 1;
              offsets[dir] = -face_sign; 
              int32_t ioffset = - ( offsets[IX] + bx_g * offsets[IY] + bx_g*by_g*offsets[IZ] );
              
              // get state in neighbor along dir
              HydroState3d qprim_n = get_prim_variables<HydroState3d>(ig + ioffset, iOct_local);

              // reconstruct "left/right" state
              HydroState3d qout =
                  reconstruct_state_3d(qprim_n, ig + ioffset, iOct_local, offsets, dtdx, dtdy, dtdz);

              // step 2 : reconstruct state in current cell (offset is negated compared to other cell)
              offsets[IX] = - offsets[IX];
              offsets[IY] = - offsets[IY];
              offsets[IZ] = - offsets[IZ];
              HydroState3d qin = reconstruct_state_3d(qprim, ig, iOct_local, offsets, dtdx, dtdy, dtdz);

              // Applying prediction step for gravity
              /*if (has_gravity) {
                apply_gravity_prediction<3>(qin, ig, iOct_local, dt);
                apply_gravity_prediction<3>(qout, ig+ioffset, iOct_local, dt);
              }*/

              HydroState3d& qL = (face==FACE_LEFT) ? qout : qin;
              HydroState3d& qR = (face==FACE_LEFT) ? qin : qout;

              VarIndex swap_component = (dir==IX) ? IU : (dir==IY) ? IV : IW;

              // riemann solver along Y or Z direction requires to 
              // swap velocity components
              swap(qL[fm_state[IU]], qL[fm_state[swap_component]]);
              swap(qR[fm_state[IU]], qR[fm_state[swap_component]]);

              // step 4 : compute flux (Riemann solver)
              HydroState3d flux = riemann_hydro(qL, qR, params.riemann_params);

              swap(flux[fm_state[IU]], flux[fm_state[swap_component]]);

              // step 5 : accumulate flux in current cell
              qcons -= flux * ( offsets[IX] * dtdx + offsets[IY] * dtdy + offsets[IZ] * dtdz ) ;
            };

            // compute from left face along x dir
            if (i > 0 or interface_flags.isFaceConformal(iOct_local, FACE_LEFT))
            {
              process_axis(IX, FACE_LEFT);
            }
            // compute flux from right face along x dir
            if (i < bx - 1 or interface_flags.isFaceConformal(iOct_local, FACE_RIGHT))
            {
              process_axis(IX, FACE_RIGHT);
            }
            // compute flux from left face along y dir
            if (j > 0 or interface_flags.isFaceConformal(iOct_local, FACE_BOTTOM))
            {
              process_axis(IY, FACE_LEFT);
            }
            // compute flux from right face along y dir
            if (j < by - 1 or interface_flags.isFaceConformal(iOct_local, FACE_TOP))
            {
              process_axis(IY, FACE_RIGHT);
            }
            // compute flux from left face along y dir
            if (k > 0 or interface_flags.isFaceConformal(iOct_local, FACE_REAR))
            {
              process_axis(IZ, FACE_LEFT);
            }
            // compute flux from right face along y dir
            if (k < bz - 1 or interface_flags.isFaceConformal(iOct_local, FACE_FRONT))
            {
              process_axis(IZ, FACE_RIGHT);
            }

            // Applying correction step for gravity
            if (has_gravity)
              apply_gravity_correction_3d(qold, qcons, ig, iOct_local, dt);
            

            // lastly update conservative variable in U2
            uint32_t index_non_ghosted = i + bx * j + bx*by *k; //(i-1) + bx * (j-1);

            U2(index_non_ghosted, fm[ID], iOct) = qcons[fm_state[ID]];
            U2(index_non_ghosted, fm[IP], iOct) = qcons[fm_state[IP]];
            U2(index_non_ghosted, fm[IU], iOct) = qcons[fm_state[IU]];
            U2(index_non_ghosted, fm[IV], iOct) = qcons[fm_state[IV]];
            U2(index_non_ghosted, fm[IW], iOct) = qcons[fm_state[IW]];

          } // end if inside inner block
        }); // end TeamVectorRange

  } // compute_fluxes_and_update_3d_conformal

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes_and_update_3d(thread2_t member) const
  {
    if (params.updateType == UPDATE_CONSERVATIVE_SUM)
    {
      //assert("3D : work in progress");
      compute_fluxes_and_update_3d_conformal(member);
      //compute_fluxes_and_update_3d_non_conformal(member);
    }
    else
    {
      assert("non conservative not implemented in 3D");
      //compute_fluxes_and_update_3d_non_conservative(member);
    }
  } // compute_fluxes_and_update_3d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const Slopes &, thread1_t member) const
  {
    if (ndim == 2)
      compute_slopes_2d(member);

    else if (ndim == 3)
      compute_slopes_3d(member);

  } // operator () - slopes

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const Fluxes &, thread2_t member) const
  {
    if (ndim == 2)
      compute_fluxes_and_update_2d(member);

    else if (ndim == 3)
      compute_fluxes_and_update_3d(member);

  } // operator () - fluxes and update

  int ndim;
  
  //! bitpit/PABLO amr mesh object
  LightOctree lmesh;

  //! general parameters
  Params params;

  //! field manager
  id2index_t fm;

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

  //! number of cells per block (used for conservative update)
  uint32_t nbCellsPerBlock;

  //! number of cells per block (used for slopes computations)
  uint32_t nbCellsPerBlock1;

  //! integer which identifies a group of octants
  uint32_t iGroup;

  //! user data for the ith group of octants
  DataArrayBlock Ugroup;

  //! user data for the entire mesh
  LegacyDataArray U;

  //! user data for the entire mesh at the end of time step
  LegacyDataArray U2;

  //! user data (primitive variables) for the ith group of octants
  DataArrayBlock Qgroup;

  //! flags at interface for 2:1 ratio
  InterfaceFlags interface_flags;

  //! time step
  real_t dt;

  //! slopes along x for current group
  DataArrayBlock SlopesX;

  //! slopes along y for current group
  DataArrayBlock SlopesY;

  //! slopes along z for current group
  DataArrayBlock SlopesZ;

  //! Should we apply gravity on the domain
  bool has_gravity;

}; // MusclBlockGodunovUpdateFunctor



} // namespace dyablo

#endif // MUSCL_BLOCK_GODUNOV_UPDATE_HYDRO_FUNCTOR_H_
