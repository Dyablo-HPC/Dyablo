#pragma once

#include "real_type.h"
#include "kokkos_shared.h"

namespace dyablo {


template<size_t dim>
using StateNd = Kokkos::Array<real_t, dim>;

// Kept for backwards compatibility this should be removed eventually
using HydroState2d = StateNd<4>;
using HydroState3d = StateNd<5>;
using GravityField = StateNd<3>;

/**
 * @brief Trait used to indicate if a State_t is convertible to a StateNd.
 * @tparam State_t the class to convert to a StateNd
 */
template< typename State_t >
struct StateNd_conversion
{
  static constexpr bool is_convertible = false;
};

template<size_t N_>
struct StateNd_conversion<StateNd<N_>> 
{
    static constexpr bool is_convertible = true;
    static constexpr size_t N = N_;
    using State_t = StateNd<N>;
    using StateNd_t = StateNd<N>;

    KOKKOS_INLINE_FUNCTION
    static const StateNd_t& to_StateNd_t(const State_t& v)
    {
        return v;
    }
    KOKKOS_INLINE_FUNCTION
    static const State_t& to_State_t(const StateNd_t& v)
    {
        return v;
    }
};

static_assert( StateNd_conversion<HydroState2d>::is_convertible );

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

template<typename State_t, 
         typename T,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true>
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

template<typename State_t, 
         typename T,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
KOKKOS_INLINE_FUNCTION
State_t operator-(const State_t& lhs_, const T& rhs)
{
    using Conv = StateNd_conversion<State_t>;
    constexpr size_t N = Conv::N;
    using StateNd_t = StateNd<N>;

    StateNd_t res{};
    StateNd_t lhs = Conv::to_StateNd_t(lhs_);
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] - rhs;
    return Conv::to_State_t(res);
}

template<typename State_t, 
         typename T,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true>
KOKKOS_INLINE_FUNCTION
State_t& operator-=(State_t &lhs, const T &rhs) {
  return lhs = lhs - rhs;
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

template<typename State_t,
         std::enable_if_t<StateNd_conversion<State_t>::is_convertible, bool> = true >
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


}