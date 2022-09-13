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

// Operators on StateNd
// Operator +
template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator+(const StateNd<N>& lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] + rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator+(const StateNd<N>& lhs, real_t rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] + rhs;
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator+(real_t lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs + rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator+=(StateNd<N>& lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        lhs[i] += rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator+=(StateNd<N>& lhs, real_t rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        lhs[i] += rhs;
    return res;
}

// Operator -
template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator-(const StateNd<N>& lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] - rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator-(const StateNd<N>& lhs, real_t rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] - rhs;
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator-(real_t lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs - rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator-=(StateNd<N>& lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        lhs[i] -= rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator-=(StateNd<N>& lhs, real_t rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        lhs[i] -= rhs;
    return res;
}

// Operator *
template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator*(const StateNd<N>& lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] * rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator*(const StateNd<N>& lhs, real_t rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] * rhs;
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator*(real_t lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs * rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator*=(StateNd<N>& lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        lhs[i] *= rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator*=(StateNd<N>& lhs, real_t rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        lhs[i] *= rhs;
    return res;
}

// Operator /
template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator/(const StateNd<N>& lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] / rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator/(const StateNd<N>& lhs, real_t rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs[i] / rhs;
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator/(real_t lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        res[i] = lhs / rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator/=(StateNd<N>& lhs, const StateNd<N>& rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        lhs[i] /= rhs[i];
    return res;
}

template< size_t N >
KOKKOS_INLINE_FUNCTION
StateNd<N> operator/=(StateNd<N>& lhs, real_t rhs)
{
    StateNd<N> res {};
    for (size_t i=0; i<N; ++i)
        lhs[i] /= rhs;
    return res;
}


}