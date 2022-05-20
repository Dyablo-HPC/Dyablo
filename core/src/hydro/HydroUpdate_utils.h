#pragma once

#include "utils_hydro.h"
#include "RiemannSolvers.h"
#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo {

/**
 * @brief Converts a given conservative variables hydro state to a 
 *        primitive hydro state 
 * 
 * @tparam ndim the number of dimensions
 * @param params hydro parameters for the conversion
 * @param Uloc the hydro state
 * @return KOKKOS_INLINE_FUNCTION 
 */
template < 
  int ndim, 
  typename PrimState,
  typename ConsState>
KOKKOS_INLINE_FUNCTION
PrimState cons_to_prim(const RiemannParams& params, const ConsState &Uloc) {
  real_t c;
  PrimState q;
  assert( ndim==3 || Uloc.rho_w == 0 );
  computePrimitives<PrimState, ConsState>(Uloc, &c, q, params.gamma0, params.smallr, params.smallp);
  
  return q;
}

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
  typename PrimState,
  typename ConsState,
  typename Array_t, 
  typename CellIndex >
KOKKOS_INLINE_FUNCTION
void compute_primitives(const RiemannParams& params, const Array_t& Ugroup, 
                        const CellIndex& iCell_Ugroup, const Array_t& Qgroup)
{
  ConsState uLoc = getConservativeState<ndim>( Ugroup, iCell_Ugroup );
  real_t c; 
  // get primitive variables in current cell
  PrimState qLoc;
  computePrimitives(uLoc, &c, qLoc, params.gamma0, params.smallr, params.smallp);

  setPrimitiveState<ndim>( Qgroup, iCell_Ugroup, qLoc );
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
  PrimState dq;

  auto slope_mm = [](real_t dvp, real_t dvm) 
  {
    if (dvp * dvm <= 0.0)
      return 0.0;
    else
      return fabs(dvp) > fabs(dvm) ? dvm : dvp;
  };

  auto slope = [&](const real_t qMinus_v, const real_t q_v, const real_t qPlus_v) 
  {
    const real_t dvp = (qPlus_v - q_v)  / dR;
    const real_t dvm = (q_v - qMinus_v) / dL;

    return slope_mm(dvp, dvm);
  };

  dq.rho = slope(qMinus.rho, q.rho, qPlus.rho);
  dq.p   = slope(qMinus.p, q.p, qPlus.p);
  dq.u   = slope(qMinus.u, q.u, qPlus.u);
  dq.v   = slope(qMinus.v, q.v, qPlus.v);
  if (ndim == 3)
    dq.w = slope(qMinus.w, q.w, qPlus.w);

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
  typename ConsState, 
  typename PrimState>
KOKKOS_INLINE_FUNCTION
ConsState compute_euler_flux( const PrimState& sourceL, const PrimState& sourceR, 
                              const PrimState& slopeL, const PrimState& slopeR,
                              ComponentIndex3D dir, 
                              const RiemannParams& params,
                              real_t dL, real_t dR)
{
  const real_t smallr = params.smallr;
  PrimState qL = reconstruct_lin_state( sourceL, slopeL,  1, smallr, dL );
  PrimState qR = reconstruct_lin_state( sourceR, slopeR, -1, smallr, dR );

  // riemann solver along Y or Z direction requires to 
  // swap velocity components
  if (dir == IY) {
    swap(qL.u, qL.v);
    swap(qR.u, qR.v);
  }
  else if (dir == IZ) {
    swap(qL.u, qL.w);
    swap(qR.u, qR.w);
  }

  // step 4 : compute flux (Riemann solver)
  ConsState flux;
  flux = riemann_hydro(qL, qR, params);

  if (dir == IY)
    swap(flux.rho_u, flux.rho_v);
  else if (dir == IZ)
    swap(flux.rho_u, flux.rho_w);
  
  return flux;
}

/**
 * @brief Calculates one Euler hydro time-step given an input. where Uout = U+dt*dU/dt
 *        and dU/dt is calculated from using a Godunov method.
 * 
 * @tparam ndim the number of dimensions 
 * @tparam CellIndex the type of index on the array U
 * @tparam PatchArray the type of patch array for the run
 * @tparam ArrayIn the type of array provided as input
 * @tparam ArrayOut the type of array provided as output
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
         typename PrimState,
         typename ConsState,
         typename CellIndex, 
         typename PatchArray, 
         typename ArrayIn,
         typename ArrayOut>
KOKKOS_INLINE_FUNCTION
void euler_update(const RiemannParams &params, 
                        ComponentIndex3D dir, 
                  const CellIndex& iCell_Uout,
                  const ArrayIn& U,
                  const PatchArray& Qgroup,
                  const real_t dt,
                  const real_t ddir,
                  const ArrayOut& Uout)
{
  typename CellIndex::offset_t offsetm = {};
  typename CellIndex::offset_t offsetp = {};
  offsetm[dir] = -1;
  offsetp[dir] = 1;

  CellIndex ib = Qgroup.convert_index(iCell_Uout);
  ConsState fluxL; 
  ConsState fluxR;
  PrimState qC = getPrimitiveState<ndim>( Qgroup, ib );
  PrimState qL = getPrimitiveState<ndim>( Qgroup, ib + offsetm );
  PrimState qR = getPrimitiveState<ndim>( Qgroup, ib + offsetp );
  PrimState qLL, qRR;
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
      const ConsState uLL = getConservativeState<ndim>( U, iUinL + offsetm );
      qLL = cons_to_prim<ndim, PrimState>(params, uLL);
    }
    else
      qLL = getPrimitiveState<ndim>( Qgroup, ib + offsetm + offsetm );
    const PrimState slopeL = compute_slope<ndim>(qLL, qL, qC, dslope_LL, dslope_L);
    
    // Central slope when looking left and right.
    // The idea here is that if you are looking right and that your left neighbor is smaller, the default 
    // SlopeC can induce a loss of conservation. For slope recreation in those cases you just use the averaged
    // value of qR (so with a cell size equal to the current one).
    // SlopeCL considers a right averaged value, SlopeCR considers a left averaged value
    const PrimState slopeCL = compute_slope<ndim>( qL, qC, qR, dslope_L, 1.0);
    fluxL = compute_euler_flux<ndim, ConsState>( qL, qC, slopeL, slopeCL, dir, params, dflux_LL, dflux_LR );
  }
  // 2- Smaller
  else {
    constexpr real_t fac = (ndim == 2 ? 0.5 : 0.25);
    foreach_smaller_neighbor<ndim>(iUinL, offsetm, U, 
                [&](const CellIndex& iCell_neighbor)
              {
                const ConsState uL = getConservativeState<ndim>( U, iCell_neighbor );
                const ConsState uLL = getConservativeState<ndim>( U, iCell_neighbor + offsetm );

                const PrimState qL = cons_to_prim<ndim, PrimState>( params, uL );
                qLL = cons_to_prim<ndim, PrimState>( params, uLL );
                const PrimState slopeC = compute_slope<ndim> ( qL, qC, qR, dslope_L, dslope_R );
                const PrimState slopeL = compute_slope<ndim> ( qLL, qL, qC, dslope_LL, dslope_L );
                fluxL += compute_euler_flux<ndim, ConsState>( qL, qC, slopeL, slopeC, dir, params, dflux_LL, dflux_LR );
              });
    fluxL *= fac; 
  }

  // == Right interface
  // 1- Same size or bigger
  if (ldiff_R >= 0) {
    if (ldiff_R > 0) {
      const ConsState uRR = getConservativeState<ndim>( U, iUinR + offsetp );
      qRR = cons_to_prim<ndim, PrimState>( params, uRR ); 
    }
    else
      qRR = getPrimitiveState<ndim>( Qgroup, ib + offsetp + offsetp );
    const PrimState slopeR = compute_slope<ndim>(qC, qR, qRR, dslope_R, dslope_RR);
    
    PrimState slopeCR = compute_slope<ndim>( qL, qC, qR, 1.0, dslope_R);
    fluxR = compute_euler_flux<ndim, ConsState>( qC, qR, slopeCR, slopeR, dir, params, dflux_RL, dflux_RR );
  }
  // 2- Smaller :
  else {
    constexpr real_t fac = (ndim == 2 ? 0.5 : 0.25);

    foreach_smaller_neighbor<ndim>(iUinR, offsetp, U, 
                [&](const CellIndex& iCell_neighbor)
              {
                const ConsState uR = getConservativeState<ndim>( U, iCell_neighbor );
                const ConsState uRR = getConservativeState<ndim>( U, iCell_neighbor + offsetp );

                const PrimState qR = cons_to_prim<ndim, PrimState>( params, uR );
                qRR = cons_to_prim<ndim, PrimState>( params, uRR );
                const PrimState slopeC = compute_slope<ndim> ( qL, qC, qR, dslope_L, dslope_R );
                const PrimState slopeR = compute_slope<ndim> ( qC, qR, qRR, dslope_R, dslope_RR );
                fluxR += compute_euler_flux<ndim, ConsState>( qC, qR, slopeC, slopeR, dir, params, dflux_RL, dflux_RR );
              });
    fluxR *= fac;      
  }

  ConsState rhs = (fluxL - fluxR) * dtddir;
  Uout.at(iCell_Uout, ID) += rhs.rho;
  Uout.at(iCell_Uout, IE) += rhs.e_tot;
  Uout.at(iCell_Uout, IU) += rhs.rho_u;
  Uout.at(iCell_Uout, IV) += rhs.rho_v;
  if(ndim==3) Uout.at(iCell_Uout, IW) += rhs.rho_w;
}

}
