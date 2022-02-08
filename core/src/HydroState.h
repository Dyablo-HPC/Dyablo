#ifndef HYDRO_STATE_H_
#define HYDRO_STATE_H_

#include "real_type.h"
#include "kokkos_shared.h"

namespace dyablo {

constexpr int HYDRO_2D_NBVAR = 4;
constexpr int HYDRO_3D_NBVAR = 5;
constexpr int MHD_2D_NBVAR = 8;
constexpr int MHD_3D_NBVAR = 8;
constexpr int MHD_NBVAR = 8;

template<size_t dim>
using StateNd = Kokkos::Array<real_t, dim>;

using HydroState2d = StateNd<HYDRO_2D_NBVAR>;
using HydroState3d = StateNd<HYDRO_3D_NBVAR>;
using GravityField = Kokkos::Array<real_t, 3>;
using MHDState = Kokkos::Array<real_t, MHD_NBVAR>;
using BField = Kokkos::Array<real_t, 3>;

// ----- 
// Operator+
// ----- 
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

} // namespace dyablo

#endif // HYDRO_STATE_H_
