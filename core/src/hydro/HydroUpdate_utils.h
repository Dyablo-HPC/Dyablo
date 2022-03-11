#pragma once

#include "utils_hydro.h"
#include "RiemannSolvers.h"
#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo {

template< int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
HydroState3d getHydroState( const Array_t& U, const CellIndex& iCell )
{
  HydroState3d res;
  res[ID] = U.at(iCell, ID);
  res[IP] = U.at(iCell, IP);
  res[IU] = U.at(iCell, IU);
  res[IV] = U.at(iCell, IV);
  if(ndim==3) res[IW] = U.at(iCell, IW);
  return res;
}

template< int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setHydroState( const Array_t& U, const CellIndex& iCell, const HydroState3d& state )
{
  U.at(iCell, ID) = state[ID];
  U.at(iCell, IP) = state[IP];
  U.at(iCell, IU) = state[IU];
  U.at(iCell, IV) = state[IV];
  if(ndim==3) U.at(iCell, IW) = state[IW];
}

template < int ndim >
KOKKOS_INLINE_FUNCTION
HydroState3d cons_to_prim(const RiemannParams& params, const HydroState3d U) {
  real_t c;
  HydroState3d q;
  computePrimitives(U, &c, q, params.gamma0, params.smallr, params.smallp);
  return q;
}

template< int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void compute_primitives(const RiemannParams& params, const Array_t& Ugroup, 
                        const CellIndex& iCell_Ugroup, const Array_t& Qgroup)
{
  HydroState3d uLoc = getHydroState<ndim>( Ugroup, iCell_Ugroup );
      
  // get primitive variables in current cell
  HydroState3d qLoc;
  real_t c = 0.0;
  if(ndim==3)
    computePrimitives(uLoc, &c, qLoc, params.gamma0, params.smallr, params.smallp);
  else
  {
    auto copy_state = [](auto& to, const auto& from){
      to[ID] = from[ID];
      to[IP] = from[IP];
      to[IU] = from[IU];
      to[IV] = from[IV];
    };
    HydroState2d uLoc_2d, qLoc_2d;
    copy_state(uLoc_2d, uLoc);
    computePrimitives(uLoc_2d, &c, qLoc_2d, params.gamma0, params.smallr, params.smallp);
    copy_state(qLoc, qLoc_2d);
  }

  setHydroState<ndim>( Qgroup, iCell_Ugroup, qLoc );
}


/// Compute slope for one cell (q_) in one direction
template< int ndim >
KOKKOS_INLINE_FUNCTION
HydroState3d compute_slope( const HydroState3d& qMinus_, 
                            const HydroState3d& q_, 
                            const HydroState3d& qPlus_, 
                            real_t dL, real_t dR)
{
  HydroState3d dq_ = {};

  auto slope_mm = [](real_t dvp, real_t dvm) 
  {
    if (dvp * dvm <= 0.0)
      return 0.0;
    else
      return fabs(dvp) > fabs(dvm) ? dvm : dvp;
  };

  auto slope = [&](VarIndex ivar) {
    const real_t dvp = (qPlus_[ivar] - q_[ivar])  / dR;
    const real_t dvm = (q_[ivar] - qMinus_[ivar]) / dL;

    dq_[ivar] = slope_mm(dvp, dvm);
  };

  slope(ID);
  slope(IP);
  slope(IU);
  slope(IV);
  if (ndim == 3)
    slope(IW);

  return dq_;
}

/**
 * Reconstructs a state from a source value and a slope.
 * \param q      a HydroState3d containing the primitive variables to be reconstructed
 * \param slope  the slope to apply
 * \param sign   the direction in which we are applying the reconstruction 
 *               (-1 left; +1 right)
 * \param smallr a minimum value for the density in case reconstruction goes negative
 * \param size   the length of the slope to apply.
 * \return a HydroState3d storing the reconstructed value
 **/
KOKKOS_INLINE_FUNCTION
HydroState3d reconstruct_lin_state( const HydroState3d& q, 
                                    const HydroState3d& slope,
                                    real_t sign, real_t smallr, real_t size )
{
  HydroState3d res = q + sign * size * slope;
  res[ID] = FMAX(res[ID], smallr);
  return res; 
}


template< int ndim >
/// Reconstruct state and apply Riemann solver at the interface between cell L and R
KOKKOS_INLINE_FUNCTION
HydroState3d compute_euler_flux( const HydroState3d& sourceL, const HydroState3d& sourceR, 
                                 const HydroState3d& slopeL, const HydroState3d& slopeR,
                                 ComponentIndex3D dir, 
                                 const RiemannParams& params,
                                 real_t dL, real_t dR)
{
  const real_t smallr = params.smallr;
  HydroState3d qL = reconstruct_lin_state( sourceL, slopeL,  1, smallr, dL );
  HydroState3d qR = reconstruct_lin_state( sourceR, slopeR, -1, smallr, dR );

  VarIndex swap_component = (dir==IX) ? IU : (dir==IY) ? IV : IW;

  // riemann solver along Y or Z direction requires to 
  // swap velocity components
  swap(qL[IU], qL[swap_component]);
  swap(qR[IU], qR[swap_component]);

  // step 4 : compute flux (Riemann solver)
  HydroState3d flux;
  if( ndim == 3 )
    flux = riemann_hydro(qL, qR, params);
  else
  {
    auto copy_state = [](auto& to, const auto& from){
      to[ID] = from[ID];
      to[IP] = from[IP];
      to[IU] = from[IU];
      to[IV] = from[IV];
    };
    HydroState2d qL_2d, qR_2d;
    copy_state(qL_2d, qL);
    copy_state(qR_2d, qR);
    HydroState2d flux_2d = riemann_hydro(qL_2d, qR_2d, params);
    copy_state(flux, flux_2d);    
  }

  swap(flux[IU], flux[swap_component]);

  return flux;
}

template<int ndim, 
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
  HydroState3d fluxL; 
  HydroState3d fluxR;
  HydroState3d qC = getHydroState<ndim>( Qgroup, ib );
  HydroState3d qL = getHydroState<ndim>( Qgroup, ib + offsetm );
  HydroState3d qR = getHydroState<ndim>( Qgroup, ib + offsetp );
  HydroState3d qLL, qRR;
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
      const HydroState3d uLL = getHydroState<ndim>( U, iUinL + offsetm );
      qLL = cons_to_prim<ndim>(params, uLL);
    }
    else
      qLL = getHydroState<ndim>( Qgroup, ib + offsetm + offsetm );
    const HydroState3d slopeL = compute_slope<ndim>(qLL, qL, qC, dslope_LL, dslope_L);
    
    // Central slope when looking left and right.
    // The idea here is that if you are looking right and that your left neighbor is smaller, the default 
    // SlopeC can induce a loss of conservation. For slope recreation in those cases you just use the averaged
    // value of qR (so with a cell size equal to the current one).
    // SlopeCL considers a right averaged value, SlopeCR considers a left averaged value
    const HydroState3d slopeCL = compute_slope<ndim>( qL, qC, qR, dslope_L, 1.0);
    fluxL = compute_euler_flux<ndim>( qL, qC, slopeL, slopeCL, dir, params, dflux_LL, dflux_LR );
  }
  // 2- Smaller
  else {
    constexpr real_t fac = (ndim == 2 ? 0.5 : 0.25);
    fluxL = HydroState3d{0};
    foreach_smaller_neighbor<ndim>(iUinL, offsetm, U, 
                [&](const CellIndex& iCell_neighbor)
              {
                const HydroState3d uL = getHydroState<ndim>( U, iCell_neighbor );
                const HydroState3d uLL = getHydroState<ndim>( U, iCell_neighbor + offsetm );

                const HydroState3d qL = cons_to_prim<ndim>( params, uL );
                qLL = cons_to_prim<ndim>( params, uLL );
                const HydroState3d slopeC = compute_slope<ndim> ( qL, qC, qR, dslope_L, dslope_R );
                const HydroState3d slopeL = compute_slope<ndim> ( qLL, qL, qC, dslope_LL, dslope_L );
                fluxL += compute_euler_flux<ndim>( qL, qC, slopeL, slopeC, dir, params, dflux_LL, dflux_LR );
              });
    fluxL *= fac; 
  }

  // == Right interface
  // 1- Same size or bigger
  if (ldiff_R >= 0) {
    if (ldiff_R > 0) {
      const HydroState3d uRR = getHydroState<ndim>( U, iUinR + offsetp );
      qRR = cons_to_prim<ndim>( params, uRR ); 
    }
    else
      qRR = getHydroState<ndim>( Qgroup, ib + offsetp + offsetp );
    const HydroState3d slopeR = compute_slope<ndim>(qC, qR, qRR, dslope_R, dslope_RR);
    
    HydroState3d slopeCR = compute_slope<ndim>( qL, qC, qR, 1.0, dslope_R);
    fluxR = compute_euler_flux<ndim>( qC, qR, slopeCR, slopeR, dir, params, dflux_RL, dflux_RR );
  }
  // 2- Smaller :
  else {
    constexpr real_t fac = (ndim == 2 ? 0.5 : 0.25);
    fluxR = HydroState3d{0};

    foreach_smaller_neighbor<ndim>(iUinR, offsetp, U, 
                [&](const CellIndex& iCell_neighbor)
              {
                const HydroState3d uR = getHydroState<ndim>( U, iCell_neighbor );
                const HydroState3d uRR = getHydroState<ndim>( U, iCell_neighbor + offsetp );

                const HydroState3d qR = cons_to_prim<ndim>( params, uR );
                qRR = cons_to_prim<ndim>( params, uRR );
                const HydroState3d slopeC = compute_slope<ndim> ( qL, qC, qR, dslope_L, dslope_R );
                const HydroState3d slopeR = compute_slope<ndim> ( qC, qR, qRR, dslope_R, dslope_RR );
                fluxR += compute_euler_flux<ndim>( qC, qR, slopeC, slopeR, dir, params, dflux_RL, dflux_RR );
              });
 
    fluxR *= fac;      
  }

  HydroState3d rhs = (fluxL - fluxR) * dtddir;
  for (auto ivar : {ID, IP, IU, IV})
    Uout.at(iCell_Uout, ivar) += rhs[ivar];
  if(ndim==3) Uout.at(iCell_Uout, IW) += rhs[IW];
}

}
