#pragma once

#include "utils_hydro.h"
#include "RiemannSolvers.h"
#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "boundary_conditions/BoundaryConditions.h"

namespace dyablo {

/**
 * @brief Computes the primitive variables at a given cell index in an array
 *        and stores it in another array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array passed to the method
 * @tparam CellIndex the type of index passed to the method
 * @param params the hydro parameters for the conversion
 * @param Ugroup The array where the conservative variables are stored
 * @param iCell_Ugroup the index where to look and where to store
 * @param Qgroup the array where the primitive variables are written
 * 
 * @note Ugroup and Qgroup should have the same sizes and properties
 */
template< 
  int ndim, 
  typename State,
  typename Array_t, 
  typename CellIndex >
KOKKOS_INLINE_FUNCTION
void compute_primitives(const RiemannParams& params,   const Array_t& Ugroup, 
                        const CellIndex& iCell_Ugroup, const Array_t& Qgroup)
{
  using ConsState = typename State::ConsState;
  using PrimState = typename State::PrimState;

  ConsState uLoc{};
  getConservativeState<ndim>(Ugroup, iCell_Ugroup, uLoc);
  PrimState qLoc = consToPrim<ndim>(uLoc, params.gamma0);
  setPrimitiveState<ndim>(Qgroup, iCell_Ugroup, qLoc);
}


/**
 * @brief Computes the slope for a given state according to given neighbors
 * 
 * @tparam ndim the number of dimensions
 * @param qMinus_ the "left" neighbor considered
 * @param q_ the current cell state
 * @param qPlus_ the "right" neighbor considered
 * @param dL the relative distance separating q_ from qMinus_
 * @param dR the relative distance separating q_ from qPlus_
 * @return the slope for each variable of q_
 * 
 * @note the relative distance is expressed in units of the size of the current cell.
 *       hence, if the neighbor has the same size, the distance will be 1.0. If the neighbor
 *       is bigger, it will be 3/2 and if it is smaller it will be 3/4.
 */
template< int ndim, typename PrimState >
KOKKOS_INLINE_FUNCTION
PrimState compute_slope( const PrimState& qMinus, 
                         const PrimState& q, 
                         const PrimState& qPlus, 
                         real_t dL, real_t dR)
{
  auto dqp = (qPlus - q)  / dR;
  auto dqm = (q - qMinus) / dL;
  
  PrimState dq{};
  state_foreach_var( [](real_t& res, real_t dvp, real_t dvm) {
    if (dvp * dvm <= 0.0)
      res = 0.0;
    else
      res = fabs(dvp) > fabs(dvm) ? dvm : dvp;
  }, dq, dqp, dqm);

  return dq;
}

/**
 * @brief Reconstructs a state from a source value and a slope.
 *
 * @param q      a PrimState containing the primitive variables to be reconstructed
 * @param slope  the slope to apply
 * @param sign   the direction in which we are applying the reconstruction 
 *               (-1 left; +1 right)
 * @param smallr a minimum value for the density in case reconstruction goes negative
 * @param size   the length of the slope to apply.
 * @return a PrimState storing the reconstructed value
 **/
template<typename PrimState>
KOKKOS_INLINE_FUNCTION
PrimState reconstruct_lin_state(const PrimState& q, 
                                const PrimState& slope,
                                real_t sign, real_t smallr, real_t size )
{
  PrimState res = q + sign * size * slope;
  res.rho = FMAX(res.rho, smallr);
  return res; 
}


/**
 * @brief Computes the flux at the interface between two cells
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param sourceL the value left of the interface
 * @param sourceR the value right of the interface
 * @param slopeL the slope left of the interface
 * @param slopeR the slope right of the interface
 * @param dir the direction of the flux 
 * @param params the hydro parameters of the run
 * @param dL the distance for the reconstruction between the left value and the interface
 * @param dR the distance for the reconstruction between the right value and the interface
 * @return the flux calculated at the interface
 */
template< 
  int ndim,
  typename State>
KOKKOS_INLINE_FUNCTION
typename State::ConsState 
compute_euler_flux(const typename State::PrimState& sourceL, 
                   const typename State::PrimState& sourceR, 
                   const typename State::PrimState& slopeL, 
                   const typename State::PrimState& slopeR,
                   ComponentIndex3D                 dir, 
                   const RiemannParams&             params,
                   real_t dL, real_t dR)
{
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  const real_t smallr = params.smallr;
  PrimState qL = reconstruct_lin_state(sourceL, slopeL,  1, smallr, dL);
  PrimState qR = reconstruct_lin_state(sourceR, slopeR, -1, smallr, dR);

  // riemann solver along Y or Z direction requires to 
  // swap velocity components
  PrimState qL_swap = swapComponents(qL, dir);
  PrimState qR_swap = swapComponents(qR, dir);

  // step 4 : compute flux (Riemann solver)
  ConsState flux{};
  flux = riemann_hydro(qL_swap, qR_swap, params);
  return swapComponents(flux, dir);
}

/**
   * @brief Computes the source term from the muscl-hancock algorithm
   *        Specialized for Hydro states
   * 
   * @tparam ndim The number of dimensions
   * @param q Centered primitive variable
   * @param slopeX Slopes along each direction
   * @param slopeY 
   * @param slopeZ 
   * @param dtdx time step over space step along each direction
   * @param dtdy 
   * @param dtdz 
   * @param gamma adiabatic index
   * @return A primitive state corresponding to the half-step evolved variable in the cell
   * 
   * @todo Adapt this to any PrimState possible !
   */
  template<int ndim >
  KOKKOS_INLINE_FUNCTION
  PrimHydroState compute_source( const PrimHydroState& q,
                                 const PrimHydroState& slopeX,
                                 const PrimHydroState& slopeY,
                                 const PrimHydroState& slopeZ,
                                 real_t dtdx, real_t dtdy, real_t dtdz,
                                 real_t gamma )
  {
    // retrieve primitive variables in current quadrant
    const real_t r = q.rho;
    const real_t p = q.p;
    const real_t u = q.u;
    const real_t v = q.v;
    const real_t w = q.w;

    // retrieve variations = dx * slopes
    const real_t drx = slopeX.rho * 0.5;
    const real_t dpx = slopeX.p   * 0.5;
    const real_t dux = slopeX.u   * 0.5;
    const real_t dvx = slopeX.v   * 0.5;
    const real_t dwx = slopeX.w   * 0.5;    
    const real_t dry = slopeY.rho * 0.5;
    const real_t dpy = slopeY.p   * 0.5;
    const real_t duy = slopeY.u   * 0.5;
    const real_t dvy = slopeY.v   * 0.5;
    const real_t dwy = slopeY.w   * 0.5;    
    const real_t drz = slopeZ.rho * 0.5;
    const real_t dpz = slopeZ.p   * 0.5;
    const real_t duz = slopeZ.u   * 0.5;
    const real_t dvz = slopeZ.v   * 0.5;
    const real_t dwz = slopeZ.w   * 0.5;

    PrimHydroState source{};
    if( ndim == 3 )
    {
      source.rho = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy + (-w * drz - dwz * r) * dtdz;
      source.u   = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy + (-w * duz) * dtdz;
      source.v   = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy + (-w * dvz) * dtdz;
      source.w   = w + (-u * dwx) * dtdx + (-v * dwy) * dtdy + (-w * dwz - dpz / r) * dtdz;
      source.p   = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy + (-w * dpz - dwz * gamma * p) * dtdz;
    }
    else
    {
      source.rho = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
      source.u   = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
      source.v   = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
      source.p   = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;
    }
    return source;
  }

  /**
   * @brief Computes the source term from the muscl-hancock algorithm
   *        Specialized for MHD states
   * 
   * @tparam ndim The number of dimensions
   * @param q Centered primitive variable
   * @param slopeX Slopes along each direction
   * @param slopeY 
   * @param slopeZ 
   * @param dtdx time step over space step along each direction
   * @param dtdy 
   * @param dtdz 
   * @param gamma adiabatic index
   * @return A primitive state corresponding to the half-step evolved variable in the cell
   * 
   * @todo Adapt this to any PrimState possible !
   */
  template<int ndim >
  KOKKOS_INLINE_FUNCTION
  PrimMHDState compute_source( const PrimMHDState& q,
                               const PrimMHDState& slopeX,
                               const PrimMHDState& slopeY,
                               const PrimMHDState& slopeZ,
                               real_t dtdx, real_t dtdy, real_t dtdz,
                               real_t gamma )
  {
    // retrieve primitive variables in current quadrant
    const real_t r = q.rho;
    const real_t p = q.p;
    const real_t u = q.u;
    const real_t v = q.v;
    const real_t w = q.w;
    const real_t Bx = q.Bx;
    const real_t By = q.By;
    const real_t Bz = q.Bz;

    // retrieve variations = dx * slopes
    const real_t drx  = slopeX.rho * 0.5;
    const real_t dpx  = slopeX.p   * 0.5;
    const real_t dux  = slopeX.u   * 0.5;
    const real_t dvx  = slopeX.v   * 0.5;
    const real_t dwx  = slopeX.w   * 0.5;
    const real_t dBxx = slopeX.Bx  * 0.5;
    const real_t dByx = slopeX.By  * 0.5;
    const real_t dBzx = slopeX.Bz  * 0.5;

    const real_t dry  = slopeY.rho * 0.5;
    const real_t dpy  = slopeY.p   * 0.5;
    const real_t duy  = slopeY.u   * 0.5;
    const real_t dvy  = slopeY.v   * 0.5;
    const real_t dwy  = slopeY.w   * 0.5;    
    const real_t dBxy = slopeY.Bx  * 0.5;    
    const real_t dByy = slopeY.By  * 0.5;    
    const real_t dBzy = slopeY.Bz  * 0.5;
    
    const real_t drz  = slopeZ.rho * 0.5;
    const real_t dpz  = slopeZ.p   * 0.5;
    const real_t duz  = slopeZ.u   * 0.5;
    const real_t dvz  = slopeZ.v   * 0.5;
    const real_t dwz  = slopeZ.w   * 0.5;    
    const real_t dBxz = slopeZ.Bx  * 0.5;    
    const real_t dByz = slopeZ.By  * 0.5;    
    const real_t dBzz = slopeZ.Bz  * 0.5;

    PrimMHDState source{};
    if( ndim == 3 )
    {
      source.rho = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy + (-w * drz - dwz * r) * dtdz;
      source.u   = u + dtdx * (-u * dux - (dpx + By*dByx + Bz*dBzx) / r) 
                     + dtdy * (-v * duy + By*dBxy/r) 
                     + dtdz * (-w * duz + Bz*dBxz/r);
      source.v   = v + dtdx * (-u * dvx + Bx*dByx/r) 
                     + dtdy * (-v * dvy - (dpy + Bx*dBxy + Bz*dBzy) / r)
                     + dtdz * (-w * dvz + Bz*dByz/r);
      source.w   = w + dtdx * (-u * dwx + Bx*dBzx/r) 
                     + dtdy * (-v * dwy + By*dBzy/r)
                     + dtdz * (-w * dwz - (dpz + Bx*dBxz + By*dByz) / r);

      source.p   = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy + (-w * dpz - dwz * gamma * p) * dtdz;
      source.Bx  = Bx + dtdy * (u*dByy + duy*By - v*dBxy - dvy*Bx)
                      + dtdz * (u*dBzz + duz*Bz - w*dBxz - dwz*Bx);
      source.By  = By + dtdx * (v*dBxx + dvx*Bx - u*dByx - dux*By)
                      + dtdz * (v*dBzz + dvz*Bz - w*dByz - dwz*By);
      source.Bz  = Bz + dtdx * (w*dBxx + dwx*Bx - u*dBzx - dux*Bz)
                      + dtdy * (w*dByy + dwy*By - v*dBzy - dvy*Bz);
    }
    else
    {
      source.rho = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
      source.u   = u + dtdx * (-u * dux - (dpx + By*dByx + Bz*dBzx) / r) 
                     + dtdy * (-v * duy + By*dBxy/r);
      source.v   = v + dtdx * (-u * dvx + Bx*dByx/r) 
                     + dtdy * (-v * dvy - (dpy + Bx*dBxy + Bz*dBzy) / r);
      source.w   = 0.0;

      source.p   = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;
      source.Bx  = Bx + dtdy * (u*dByy + duy*By - v*dBxy - dvy*Bx);
      source.By  = By + dtdx * (v*dBxx + dvx*Bx - u*dByx - dux*By);
      source.Bz  = Bz + dtdx * (-u*dBzx -dux*Bz)
                      + dtdy * (-v*dBzy -dvy*Bz);
    }
    return source;
  }


/**
 * @brief Calculates one Euler hydro time-step given an input. where Uout = U+dt*dU/dt
 *        and dU/dt is calculated using a Godunov method.
 * 
 * @tparam ndim the number of dimensions 
 * @tparam CellIndex the type of index on the array U
 * @tparam PatchArray the type of patch array for the run
 * @tparam ArrayIn the type of array provided as input
 * @tparam ArrayOut the type of array provided as output
 * 
 * @param params the hydro parameters of the run
 * @param dir the direction for the update
 * @param iCell_Uout the index of the cell where the results will be written
 * @param U the input array
 * @param Qgroup a ghosted array holding primitive variables
 * @param dt the timestep
 * @param ddir the size of the current cell
 * @param Uout the output array 
 */
template<int ndim, 
         typename State,
         typename CellIndex, 
         typename PatchArray, 
         typename ArrayIn,
         typename ArrayOut>
KOKKOS_INLINE_FUNCTION
void euler_update(const RiemannParams&     params, 
                        ComponentIndex3D   dir, 
                  const CellIndex&         iCell_Uout,
                  const ArrayIn&           U,
                  const PatchArray&        Qgroup,
                  const real_t             dt,
                  const real_t             ddir,
                  const BoundaryConditions bc_manager,
                  const ArrayOut&          Uout)
{
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  typename CellIndex::offset_t offsetm = {};
  typename CellIndex::offset_t offsetp = {};
  offsetm[dir] = -1;
  offsetp[dir] = 1;

  CellIndex ib = Qgroup.convert_index(iCell_Uout);
  ConsState fluxL{}; 
  ConsState fluxR{};
  PrimState qC{}, qL{}, qR{};
  getPrimitiveState<ndim>(Qgroup, ib, qC);
  getPrimitiveState<ndim>(Qgroup, ib + offsetm, qL);
  getPrimitiveState<ndim>(Qgroup, ib + offsetp, qR);
  PrimState qLL{}, qRR{};
  const real_t dtddir = dt / ddir;

  // Getting the current cell in U
  // U and Uout don't necessarily have the same size so we have to do this
  CellIndex iin = U.convert_index(iCell_Uout);

  // Left and right cells and level diffs
  CellIndex iUinL = iin.getNeighbor_ghost(offsetm, U);
  CellIndex iUinR = iin.getNeighbor_ghost(offsetp, U);
  int ldiff_L = iUinL.level_diff();
  int ldiff_R = iUinR.level_diff();

  /**
   *  Slope and fluxes distances with AMR see dyablo doc sec.3.3
   *
   * . ds_factors is the distance between the cell centers along the direction, 
   *              in units of the current cell size. The table has three couples
   *              of values. Each couple corresponds to a level_diff() value
   *              and in the couple (A, B), A corresponds to the distance between
   *              the two furthest points, and B between the center of the current
   *              cell and the next.
   * . df_factors is the distance between the cell center and the interface along
   *              As for ds_factors it is scaled in the units of the current cell size.
   *              The first value of the couple represents the distance between the 
   *              cell center of the next cell across the boundary and the interface.
   *              The second value is the distance between the current cell center
   *              and the interface  
   * 
   * Use cases : 
   * 1- Same size neighbors (ds/df_factors[1]) :
   *                
   *       |<----->| ds[1][1]
   *               |<----->| ds[1][0]
   *   +-------+-------+-------+
   *   |       |       |       |
   *   |   x   |   x   |   x   |
   *   |       |       |       |
   *   +-------+-------+-------+
   *       |<->| df[1][1]
   *           |<->| df[1][0]
   * 
   * 2- Smaller neighbors (ds/df_factors[0])
   * 
   *       |<--->| ds[0][1]
   *             |<->| ds[0][0]
   *   +-------+---+---+
   *   |       | x | x |
   *   |   x   +---+---+
   *   |       | x | x |
   *   +-------+---+---+
   *       |<->| df[0][1]
   *           |-| df[0][0]
   * 
   * 3- Larger neighbors (ds/df_factors[2])
   *   
   * 
   *       |<--------->| ds[2][1]
   *                   |<------------->| ds[2][0]
   *   +-------+---------------+---------------+
   *   |       |               |               |
   *   |   x   |               |               |
   *   |       |               |               |
   *   +-------+       x       |       x       |
   *   |       |               |               |
   *   |   x   |               |               |
   *   |       |               |               |
   *   +-------+---------------+---------------+
   *       |<->| df[2][1]
   *           |<----->| df[2][0]
   **/
  const real_t ds_factors[3][2] {{0.5, 0.75}, {1.0, 1.0}, {2.0, 1.5}};
  const real_t df_factors[3][2] {{0.25, 0.5}, {0.5, 0.5}, {1.0, 0.5}};
  real_t dslope_LL = ds_factors[ldiff_L+1][0];
  real_t dslope_L  = ds_factors[ldiff_L+1][1];
  real_t dslope_RR = ds_factors[ldiff_R+1][0];
  real_t dslope_R  = ds_factors[ldiff_R+1][1];
  real_t dflux_LL  = df_factors[ldiff_L+1][0];
  real_t dflux_LR  = df_factors[ldiff_L+1][1];
  real_t dflux_RL  = df_factors[ldiff_R+1][1]; // this is not an error
  real_t dflux_RR  = df_factors[ldiff_R+1][0];
  
  // == Left interface
  // 1- Same size or bigger
  if (ldiff_L >= 0) {
    if (ldiff_L > 0) {
      ConsState uLL{};
      getConservativeState<ndim>(U, iUinL + offsetm, uLL);
      qLL = consToPrim<ndim>(uLL, params.gamma0);
    }
    else 
      getPrimitiveState<ndim>(Qgroup, ib + offsetm + offsetm, qLL);

    const PrimState slopeL = compute_slope<ndim>(qLL, qL, qC, dslope_LL, dslope_L);
    
    // Central slope when looking left and right.
    // The idea here is that if you are looking right and that your left neighbor is smaller, the default 
    // SlopeC can induce a loss of conservation. For slope recreation in those cases you just use the averaged
    // value of qR (so with a cell size equal to the current one).
    // SlopeCL considers a right averaged value, SlopeCR considers a left averaged value
    const PrimState slopeCL = compute_slope<ndim>(qL, qC, qR, dslope_L, 1.0);
    fluxL = compute_euler_flux<ndim, State>(qL, qC, slopeL, slopeCL, dir, params, dflux_LL, dflux_LR);

    if (iUinL.is_boundary() && bc_manager.bc_min[dir] == BC_USER)
      fluxL = bc_manager.template overrideBoundaryFlux<ndim, State>(fluxL, qC, dir, true);
  }
  // 2- Smaller
  else {
    constexpr real_t fac = (ndim == 2 ? 0.5 : 0.25);
    foreach_smaller_neighbor<ndim>(iUinL, offsetm, U, 
                [&](const CellIndex& iCell_neighbor)
              {
                ConsState uL{}, uLL{};
                getConservativeState<ndim>(U, iCell_neighbor, uL);
                getConservativeState<ndim>(U, iCell_neighbor + offsetm, uLL);

                const PrimState qL = consToPrim<ndim>(uL, params.gamma0);
                qLL = consToPrim<ndim>(uLL, params.gamma0);
                const PrimState slopeC = compute_slope<ndim>(qL, qC, qR, dslope_L, dslope_R);
                const PrimState slopeL = compute_slope<ndim>(qLL, qL, qC, dslope_LL, dslope_L);
                fluxL += compute_euler_flux<ndim, State>(qL, qC, slopeL, slopeC, dir, params, dflux_LL, dflux_LR);
              });
    fluxL *= fac; 
  }

  // == Right interface
  // 1- Same size or bigger
  if (ldiff_R >= 0) {
    if (ldiff_R > 0) {
      ConsState uRR{};
      getConservativeState<ndim>(U, iUinR + offsetp, uRR);
      qRR = consToPrim<ndim>(uRR, params.gamma0); 
    }
    else
      getPrimitiveState<ndim>(Qgroup, ib + offsetp + offsetp, qRR);

    const PrimState slopeR = compute_slope<ndim>(qC, qR, qRR, dslope_R, dslope_RR);
    
    PrimState slopeCR = compute_slope<ndim>(qL, qC, qR, 1.0, dslope_R);
    fluxR = compute_euler_flux<ndim, State>(qC, qR, slopeCR, slopeR, dir, params, dflux_RL, dflux_RR);

    if (iUinR.is_boundary() && bc_manager.bc_max[dir] == BC_USER)
      fluxR = bc_manager.template overrideBoundaryFlux<ndim, State>(fluxR, qC, dir, false);
  }
  // 2- Smaller :
  else {
    constexpr real_t fac = (ndim == 2 ? 0.5 : 0.25);

    foreach_smaller_neighbor<ndim>(iUinR, offsetp, U, 
                [&](const CellIndex& iCell_neighbor)
              {
                ConsState uR{}, uRR{};
                getConservativeState<ndim>(U, iCell_neighbor, uR);
                getConservativeState<ndim>(U, iCell_neighbor + offsetp, uRR);

                const PrimState qR = consToPrim<ndim>(uR, params.gamma0);
                qRR = consToPrim<ndim>(uRR, params.gamma0);
                const PrimState slopeC = compute_slope<ndim>(qL, qC, qR,  dslope_L, dslope_R);
                const PrimState slopeR = compute_slope<ndim>(qC, qR, qRR, dslope_R, dslope_RR);
                fluxR += compute_euler_flux<ndim, State>(qC, qR, slopeC, slopeR, dir, params, dflux_RL, dflux_RR);
              });
    fluxR *= fac;      
  }

  ConsState u{};
  getConservativeState<ndim>(Uout, iCell_Uout, u);
  u += (fluxL - fluxR) * dtddir;
  setConservativeState<ndim>(Uout, iCell_Uout, u);
}

}
