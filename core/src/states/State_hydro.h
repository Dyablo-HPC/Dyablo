#pragma once

#include "real_type.h"
#include "kokkos_shared.h"
#include "State_Nd.h"
#include "FieldManager.h"

namespace dyablo {


/**
 * @brief Structure holding conservative hydrodynamics variables
 **/ 
struct ConsHydroState {
  real_t rho = 0;
  real_t e_tot = 0;
  real_t rho_u = 0;
  real_t rho_v = 0;
  real_t rho_w = 0;
};

/**
 * @brief StateNd_conversion for ConsHydroState
 */
template<>
struct StateNd_conversion<ConsHydroState> 
{
    static constexpr bool is_convertible = true;
    static constexpr size_t N = 5;
    using State_t = ConsHydroState;
    using StateNd_t = StateNd<N>;

    KOKKOS_INLINE_FUNCTION
    static StateNd_t to_StateNd_t(const State_t& v)
    {
        return {v.rho, v.e_tot, v.rho_u, v.rho_v, v.rho_w};
    }
    KOKKOS_INLINE_FUNCTION
    static State_t to_State_t(const StateNd_t& v)
    {
        return {v[0],v[1],v[2],v[3],v[4]};
    }
};


/**
 * @brief Structure holding primitive hydrodynamics variables
 */
struct PrimHydroState {
  real_t rho = 0;
  real_t p = 0;
  real_t u = 0;
  real_t v = 0;
  real_t w = 0;
};

/**
 * @brief StateNd_conversion for PrimHydroState  
 */
template<>
struct StateNd_conversion<PrimHydroState> 
{
    static constexpr bool is_convertible = true;
    static constexpr size_t N = 5;
    using State_t = PrimHydroState;
    using StateNd_t = StateNd<N>;

    KOKKOS_INLINE_FUNCTION
    static StateNd_t to_StateNd_t(const State_t& v)
    {
        return {v.rho, v.p, v.u, v.v, v.w};
    }
    KOKKOS_INLINE_FUNCTION
    static State_t to_State_t(const StateNd_t& v)
    {
        return {v[0],v[1],v[2],v[3],v[4]};
    }
};

struct HydroState {
  using PrimState = PrimHydroState;
  using ConsState = ConsHydroState;
  static constexpr size_t N = 5;
};

/**
 * @brief Returns a conservative state at a given cell index in an array
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
void getConservativeState(const Array_t& U, const CellIndex& iCell, ConsHydroState &res)
{
  res.rho   = U.at(iCell, ID);
  res.e_tot = U.at(iCell, IE);
  res.rho_u = U.at(iCell, IU);
  res.rho_v = U.at(iCell, IV);
  res.rho_w = (ndim == 3 ? U.at(iCell, IW) : 0.0);
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
void getPrimitiveState(const Array_t& U, const CellIndex& iCell, PrimHydroState &res)
{
  res.rho = U.at(iCell, ID);
  res.p   = U.at(iCell, IP);
  res.u   = U.at(iCell, IU);
  res.v   = U.at(iCell, IV);
  res.w   = (ndim == 3 ? U.at(iCell, IW) : 0.0);
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
void setPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimHydroState u) {
  U.at(iCell, ID) = u.rho;
  U.at(iCell, IP) = u.p;
  U.at(iCell, IU) = u.u;
  U.at(iCell, IV) = u.v;
  if (ndim == 3)
    U.at(iCell, IW) = u.w;
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
void setConservativeState( const Array_t& U, const CellIndex& iCell, ConsHydroState u) {
  U.at(iCell, ID) = u.rho;
  U.at(iCell, IE) = u.e_tot;
  U.at(iCell, IU) = u.rho_u;
  U.at(iCell, IV) = u.rho_v;
  if (ndim == 3)
    U.at(iCell, IW) = u.rho_w;
}

/**
 * @brief Converts from a hydro conservative state to a hydro primitive state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param U the initial conservative state
 * @param gamma0 adiabatic index
 * @return the primitive version of U
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
PrimHydroState consToPrim(const ConsHydroState &U, real_t gamma0) {
  const real_t Ek = 0.5 * (U.rho_u*U.rho_u+U.rho_v*U.rho_v+U.rho_w*U.rho_w)/U.rho;
  const real_t p = (U.e_tot - Ek) * (gamma0-1.0);
  return {U.rho, 
          p, 
          U.rho_u/U.rho, 
          U.rho_v/U.rho, 
          (ndim == 3 ? U.rho_w/U.rho : 0.0)};
}

/**
 * @brief Converts from a hydro primitive state to a hydro conservative state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param Q the initial primitive state
 * @param gamma0 adiabatic index
 * @return the conservative version of Q
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
ConsHydroState primToCons(const PrimHydroState &Q, real_t gamma0) {
    const real_t Ek = 0.5 * Q.rho * (Q.u*Q.u+Q.v*Q.v+Q.w*Q.w);
    const real_t E  = Ek + Q.p / (gamma0-1.0);
    return {Q.rho, 
            E, 
            Q.rho*Q.u, 
            Q.rho*Q.v, 
            (ndim ==3 ? Q.rho*Q.w : 0.0)};
}

} // namespace dyablo

