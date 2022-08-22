#pragma once

#include "real_type.h"
#include "kokkos_shared.h"
#include "State_Ops.h"

namespace dyablo {

/**
 * @brief Structure holding conservative magneto-hydrodynamics variables
 **/ 
struct ConsMHDState {
  real_t rho = 0;
  real_t e_tot = 0;
  real_t rho_u = 0;
  real_t rho_v = 0;
  real_t rho_w = 0;
  real_t Bx = 0;
  real_t By = 0;
  real_t Bz = 0;
};

DECLARE_STATE_TYPE( ConsMHDState, 8 );
DECLARE_STATE_GET( ConsMHDState, 0, rho );
DECLARE_STATE_GET( ConsMHDState, 1, e_tot );
DECLARE_STATE_GET( ConsMHDState, 2, rho_u );
DECLARE_STATE_GET( ConsMHDState, 3, rho_v );
DECLARE_STATE_GET( ConsMHDState, 4, rho_w );
DECLARE_STATE_GET( ConsMHDState, 5, Bx );
DECLARE_STATE_GET( ConsMHDState, 6, By );
DECLARE_STATE_GET( ConsMHDState, 7, Bz );

/**
 * @brief Structure holding primitive magneto-hydrodynamics variables
 */
struct PrimMHDState {
  real_t rho = 0;
  real_t p = 0;
  real_t u = 0;
  real_t v = 0;
  real_t w = 0;
  real_t Bx = 0;
  real_t By = 0;
  real_t Bz = 0;
};

DECLARE_STATE_TYPE( PrimMHDState, 8 );
DECLARE_STATE_GET( PrimMHDState, 0, rho );
DECLARE_STATE_GET( PrimMHDState, 1, p );
DECLARE_STATE_GET( PrimMHDState, 2, u );
DECLARE_STATE_GET( PrimMHDState, 3, v );
DECLARE_STATE_GET( PrimMHDState, 4, w );
DECLARE_STATE_GET( PrimMHDState, 5, Bx );
DECLARE_STATE_GET( PrimMHDState, 6, By );
DECLARE_STATE_GET( PrimMHDState, 7, Bz );



/**
 * @brief Structure grouping the primitive and conservative MHD state as well
 *        as information on the number of fields to store per state
 */
struct MHDState {
  using PrimState = PrimMHDState;
  using ConsState = ConsMHDState;
  static constexpr size_t N = 8;
};

/**
* @brief Returns a conservative state at a given cell index in an array
* 
* @tparam ndim the number of dimensions
* @tparam Array_t the type of array where we are looking up
* @tparam CellIndex the type of cell index used

* @param U the array in which we are getting the state
* @param iCell the index of the cell 
* @return the hydro state at position iCell in U
*/
template< int ndim, 
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getConservativeState( const Array_t& U, const CellIndex& iCell, ConsMHDState &res )
{
  res.rho   = U.at(iCell, ID);
  res.e_tot = U.at(iCell, IE);
  res.rho_u = U.at(iCell, IU);
  res.rho_v = U.at(iCell, IV);
  res.rho_w = (ndim == 3 ? U.at(iCell, IW) : 0.0);
  res.Bx = U.at(iCell, IBX);
  res.By = U.at(iCell, IBY);
  res.Bz = U.at(iCell, IBZ);
}

/**
 * @brief Returns a primitive hydro state at a given cell index in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array where we are looking up
 * @tparam CellIndex the type of cell index used
 *
 * @param U the array in which we are getting the state
 * @param iCell the index of the cell 
 * @return the hydro state at position iCell in U
 */
template< int ndim,
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimMHDState &res )
{
  res.rho = U.at(iCell, ID);
  res.p   = U.at(iCell, IP);
  res.u   = U.at(iCell, IU);
  res.v   = U.at(iCell, IV);
  res.w   = (ndim == 3 ? U.at(iCell, IW) : 0.0);
  res.Bx  = U.at(iCell, IBX);
  res.By  = U.at(iCell, IBY);
  res.Bz  = U.at(iCell, IBZ);
}

/**
 * @brief Stores a primitive hydro state in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array in which the primitive value is stored
 * @tparam CellIndex the type of cell index used
 * 
 * @param U the array where we are storing the state
 * @param iCell the index of cell
 * @param u the value to store in the array
 */
template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimMHDState u) {
  U.at(iCell, ID) = u.rho;
  U.at(iCell, IP) = u.p;
  U.at(iCell, IU) = u.u;
  U.at(iCell, IV) = u.v;
  U.at(iCell, IBX) = u.Bx;
  U.at(iCell, IBY) = u.By;
  U.at(iCell, IBZ) = u.Bz;
  if (ndim == 3) {
    U.at(iCell, IW) = u.w;
  }
}

/**
 * @brief Stores a conservative hydro state in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array in which the primitive value is stored
 * @tparam CellIndex the type of cell index used
 * 
 * @param U the array where we are storing the state
 * @param iCell the index of cell
 * @param u the value to store in the array
 */
template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setConservativeState( const Array_t& U, const CellIndex& iCell, ConsMHDState u) {
  U.at(iCell, ID) = u.rho;
  U.at(iCell, IE) = u.e_tot;
  U.at(iCell, IU) = u.rho_u;
  U.at(iCell, IV) = u.rho_v;
  U.at(iCell, IBX) = u.Bx;
  U.at(iCell, IBY) = u.By;
  U.at(iCell, IBZ) = u.Bz;
  if (ndim == 3) {
    U.at(iCell, IW) = u.rho_w;
  }
}

/**
 * @brief Converts from a MHD conservative state to a MHD primitive state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param U the initial conservative state
 * @param gamma0 adiabatic index
 * @return the primitive version of U
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
PrimMHDState consToPrim(const ConsMHDState &U, real_t gamma0) {
  const real_t Ek = 0.5 * (U.rho_u*U.rho_u
                          +U.rho_v*U.rho_v
                          +(ndim == 3 ? U.rho_w*U.rho_w : 0.0))/U.rho;

  const real_t Em = 0.5 * (U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz);
  const real_t p = (U.e_tot - Ek - Em) * (gamma0-1.0);
  return {U.rho, 
          p, 
          U.rho_u/U.rho, 
          U.rho_v/U.rho, 
          (ndim == 3 ? U.rho_w/U.rho : 0.0),
          U.Bx,
          U.By,
          U.Bz};
}

/**
 * @brief Converts from a MHD primitive state to a MHD conservative state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param Q the initial primitive state
 * @param gamma0 adiabatic index
 * @return the conservative version of Q
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
ConsMHDState primToCons(const PrimMHDState &Q, real_t gamma0) {
  const real_t Ek = 0.5 * Q.rho * (Q.u*Q.u
                                  +Q.v*Q.v
                                  +(ndim == 3 ? Q.w*Q.w : 0.0));
                                  
  const real_t Em = 0.5 * (Q.Bx*Q.Bx + Q.By*Q.By + Q.Bz*Q.Bz);
  const real_t E  = Ek + Em + Q.p / (gamma0-1.0);
  return {Q.rho, 
          E, 
          Q.rho*Q.u, 
          Q.rho*Q.v, 
          (ndim ==3 ? Q.rho*Q.w : 0.0),
          Q.Bx,
          Q.By,
          Q.Bz};
}

/**
 * @brief Swaps a component in velocity and magnetic field with the X component. 
 *        The Riemann problem is always solved by considering an interface on the 
 *        X-axis. So when solving it for other components, those should be swapped 
 *        before and after solving the Riemann problem.
 *  
 * @param Q (IN/OUT) the primitive MHD state to modify
 * @param comp the component to swap with X
 */
KOKKOS_INLINE_FUNCTION
PrimMHDState swapComponents(const PrimMHDState &q, ComponentIndex3D comp) {
  switch( comp )
  {
    case IX:
      return q;
    case IY:
      return PrimMHDState{q.rho, q.p, q.v, q.u, q.w, q.By, q.Bx, q.Bz};
    case IZ:
      return PrimMHDState{q.rho, q.p, q.w, q.v, q.u, q.Bz, q.By, q.Bx};
    default:
      assert(false);
      return PrimMHDState{};
  }
}

/**
 * @brief Swaps a component in velocity and magnetic field with the X component. 
 *        The Riemann problem is always solved by considering an interface on the 
 *        X-axis. So when solving it for other components, those should be swapped 
 *        before and after solving the Riemann problem.
 *  
 * @param Q (IN/OUT) the primitive MHD state to modify
 * @param comp the component to swap with X
 */
KOKKOS_INLINE_FUNCTION
ConsMHDState swapComponents(const ConsMHDState &u, ComponentIndex3D comp) {
  switch( comp )
  {
    case IX:
      return u;
    case IY:
      return ConsMHDState{u.rho, u.e_tot, u.rho_v, u.rho_u, u.rho_w, u.By, u.Bx, u.Bz};
    case IZ:
      return ConsMHDState{u.rho, u.e_tot, u.rho_w, u.rho_v, u.rho_u, u.Bz, u.By, u.Bx};
    default:
      assert(false);
      return ConsMHDState{};
  }
}
} // namespace dyablo

