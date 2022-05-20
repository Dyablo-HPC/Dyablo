#pragma once

#include "real_type.h"
#include "kokkos_shared.h"

namespace dyablo {

template<size_t dim>
using StateNd = Kokkos::Array<real_t, dim>;

// Kept for backwards compatibility this should be removed eventually
using HydroState2d = StateNd<4>;
using HydroState3d = StateNd<5>;



using GravityField = Kokkos::Array<real_t, 3>;

/** 
 * Direct StateNd operations
 * NOTE : Kept here for backwards compatibility.
 *        Those are used in Kokkos::Array<double> operations
 * Eventually, these should maybe moved to another place where we store all the
 * aliases such as "offset_t", "pos_t", etc.
 **/
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator+(const StateNd<dim>& lhs, const StateNd<dim>& rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs[i] + rhs[i];
  return res;
}
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator+(const StateNd<dim>& lhs, real_t rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs[i] + rhs;
  return res;
}
template<size_t dim, typename T>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator+( const T& lhs, const StateNd<dim>& rhs)
{
  return rhs + lhs;
}
template<size_t dim, typename T>
KOKKOS_INLINE_FUNCTION
StateNd<dim>& operator+=(StateNd<dim>& lhs, const T& rhs)
{
  return lhs = lhs + rhs;
}

// ----- 
// Operator-
// ----- 
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator-(const StateNd<dim>& lhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = -lhs[i];
  return res;
}
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator-(const StateNd<dim>& lhs, const StateNd<dim>& rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs[i] - rhs[i];
  return res;
}
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator-(const StateNd<dim>& lhs, real_t rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs[i] - rhs;
  return res;
}
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator-( real_t lhs, const StateNd<dim>& rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs - rhs[i];
  return res;
}
template<size_t dim, typename T>
KOKKOS_INLINE_FUNCTION
StateNd<dim>& operator-=(StateNd<dim>& lhs, const T& rhs)
{
  return lhs = lhs - rhs;
}

// ----- 
// Operator*
// ----- 
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator*(const StateNd<dim>& lhs, const StateNd<dim>& rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs[i] * rhs[i];
  return res;
}
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator*(const StateNd<dim>& lhs, real_t rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs[i] * rhs;
  return res;
}
template<size_t dim, typename T>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator*( const T& lhs, const StateNd<dim>& rhs)
{
  return rhs * lhs;
}
template<size_t dim, typename T>
KOKKOS_INLINE_FUNCTION
StateNd<dim>& operator*=(StateNd<dim>& lhs, const T& rhs)
{
  return lhs = lhs * rhs;
}

// -----
// Operator/
// -----
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator/(const StateNd<dim>& lhs, const StateNd<dim>& rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs[i] / rhs[i];
  return res;
}
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator/(const StateNd<dim>& lhs, real_t rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs[i] / rhs;
  return res;
}
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim> operator/(real_t lhs, const StateNd<dim>& rhs)
{
  StateNd<dim> res{};
  for (size_t i=0; i<dim; ++i)
    res[i] = lhs / rhs[i];
  return res;
}
template<size_t dim, typename T>
KOKKOS_INLINE_FUNCTION
StateNd<dim>& operator/=(StateNd<dim>& lhs, const T& rhs)
{
  return lhs = lhs / rhs;
}

/**
 * @brief Trait used to indicate if a State_t is convertible to a StateNd.
 * @tparam State_t the class to convert to a StateNd
 */
template< typename State_t >
struct StateNd_conversion
{
  static constexpr bool is_convertible = false;
};

/**
 * @brief Structure holding conservative hydrodynamics variables
 **/ 
struct ConsHydroState {
  real_t rho;
  real_t e_tot;
  real_t rho_u;
  real_t rho_v;
  real_t rho_w;

  KOKKOS_INLINE_FUNCTION
  ConsHydroState() : rho(0.0), e_tot(0.0), rho_u(0.0), rho_v(0.0), rho_w(0.0) {};

  KOKKOS_INLINE_FUNCTION
  ConsHydroState(real_t rho, real_t e_tot, real_t rho_u, real_t rho_v, real_t rho_w)
    : rho(rho), e_tot(e_tot), rho_u(rho_u), rho_v(rho_v), rho_w(rho_w) {};
};

/**
 * @brief Structure holding primitive hydrodynamics variables
 */
struct PrimHydroState {
  real_t rho;
  real_t p;
  real_t u;
  real_t v;
  real_t w;

  KOKKOS_INLINE_FUNCTION
  PrimHydroState() : rho(0.0), p(0.0), u(0.0), v(0.0), w(0.0) {};
  
  KOKKOS_INLINE_FUNCTION
  PrimHydroState(real_t rho, real_t p, real_t u, real_t v, real_t w)
    : rho(rho), p(p), u(u), v(v), w(w) {};
};

/**
 * @brief Converts a conservative hydro state to and from a StateNd
 */
template<>
struct StateNd_conversion<ConsHydroState> 
{
    static constexpr bool is_convertible = true;
    static constexpr size_t N = 5;
    using State_t = ConsHydroState;
    using StateNd_t = StateNd<N>;

    static_assert( sizeof(State_t) == N*sizeof(real_t) );

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
 * @brief Converts a primitive hydro state to and from a StateNd  
 */
template<>
struct StateNd_conversion<PrimHydroState> 
{
    static constexpr bool is_convertible = true;
    static constexpr size_t N = 5;
    using State_t = PrimHydroState;
    using StateNd_t = StateNd<N>;

    static_assert( sizeof(State_t) == N*sizeof(real_t) );

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

// Operators on convertible states
// Operator +
template<typename State_t,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t operator+(const State_t& lhs_, const State_t& rhs_)
{
    using Conv = StateNd_conversion<State_t>;
    constexpr size_t N = Conv::N;
    using StateNd_t = StateNd<N>;

    StateNd_t res{};
    StateNd_t lhs = Conv::to_StateNd_t(lhs_);
    StateNd_t rhs = Conv::to_StateNd_t(rhs_);
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] + rhs[i];
    return Conv::to_State_t(res);
}

template<typename State_t, 
         typename T,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t operator+(const State_t& lhs_, const T& rhs)
{
    using Conv = StateNd_conversion<State_t>;
    constexpr size_t N = Conv::N;
    using StateNd_t = StateNd<N>;

    StateNd_t res{};
    StateNd_t lhs = Conv::to_StateNd_t(lhs_);
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] + rhs;
    return Conv::to_State_t(res);
}

template<typename State_t, typename T>
KOKKOS_INLINE_FUNCTION
State_t& operator+=(State_t &lhs, const T &rhs) {
  return lhs = lhs + rhs;
}

// Operator -
template<typename State_t,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t operator-(const State_t& lhs_, const State_t& rhs_)
{
    using Conv = StateNd_conversion<State_t>;
    constexpr size_t N = Conv::N;
    using StateNd_t = StateNd<N>;

    StateNd_t res{};
    StateNd_t lhs = Conv::to_StateNd_t(lhs_);
    StateNd_t rhs = Conv::to_StateNd_t(rhs_);
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] - rhs[i];
    return Conv::to_State_t(res);
}

// Operator *
template<typename State_t,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t operator*(const State_t& lhs_, const State_t& rhs_)
{
    using Conv = StateNd_conversion<State_t>;
    constexpr size_t N = Conv::N;
    using StateNd_t = StateNd<N>;

    StateNd_t res{};
    StateNd_t lhs = Conv::to_StateNd_t(lhs_);
    StateNd_t rhs = Conv::to_StateNd_t(rhs_);
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] * rhs[i];
    return Conv::to_State_t(res);
}

template<typename State_t, 
         typename T,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t operator*(const State_t& lhs_, const T& rhs)
{
    using Conv = StateNd_conversion<State_t>;
    constexpr size_t N = Conv::N;
    using StateNd_t = StateNd<N>;

    StateNd_t res{};
    StateNd_t lhs = Conv::to_StateNd_t(lhs_);
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] * rhs;
    return Conv::to_State_t(res);
}

template<typename State_t, 
         typename T,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t operator*(const T& lhs, const State_t& rhs)
{
  return rhs*lhs;
}

template<typename State_t>
KOKKOS_INLINE_FUNCTION
State_t& operator*=(State_t &lhs, const real_t &rhs) {
  return lhs = lhs * rhs;
}


// Operator /
template<typename State_t,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t operator/(const State_t& lhs_, const real_t rhs_)
{
    using Conv = StateNd_conversion<State_t>;
    constexpr size_t N = Conv::N;
    using StateNd_t = StateNd<N>;

    StateNd_t res{};
    StateNd_t lhs = Conv::to_StateNd_t(lhs_);
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] / rhs_;
    return Conv::to_State_t(res);
}

template<typename State_t,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t& operator/=(State_t &lhs, const real_t &rhs) {
  return lhs = lhs / rhs;
}

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
ConsHydroState getConservativeState( const Array_t& U, const CellIndex& iCell )
{
  ConsHydroState res;
  res.rho   = U.at(iCell, ID);
  res.e_tot = U.at(iCell, IE);
  res.rho_u = U.at(iCell, IU);
  res.rho_v = U.at(iCell, IV);
  res.rho_w = (ndim == 3 ? U.at(iCell, IW) : 0.0);
  return res;
}

/**
 * @brief Returns a primitive hydro state at a given cell index in an array
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
PrimHydroState getPrimitiveState( const Array_t& U, const CellIndex& iCell )
{
  PrimHydroState res;
  res.rho = U.at(iCell, ID);
  res.p   = U.at(iCell, IP);
  res.u   = U.at(iCell, IU);
  res.v   = U.at(iCell, IV);
  res.w   = (ndim == 3 ? U.at(iCell, IW) : 0.0);
  return res;
}

/**
 * @brief Stores a primitive hydro state in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array in which the primitive value is stored
 * @tparam CellIndex the type of cell index used
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

} // namespace dyablo

