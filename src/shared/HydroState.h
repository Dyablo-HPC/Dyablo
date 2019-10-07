#ifndef HYDRO_STATE_H_
#define HYDRO_STATE_H_

#include "real_type.h"

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
using MHDState = Kokkos::Array<real_t, MHD_NBVAR>;
using BField = Kokkos::Array<real_t, 3>;

// =================================================================
// =================================================================
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim>& operator+=(StateNd<dim>& lhs, const StateNd<dim>& rhs)
{

  for (size_t i=0; i<dim; ++i)
    lhs[i] += rhs[i];

  return lhs;

} // operator+=

// =================================================================
// =================================================================
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim>& operator-=(StateNd<dim>& lhs, const StateNd<dim>& rhs)
{

  for (size_t i=0; i<dim; ++i)
    lhs[i] -= rhs[i];

  return lhs;

} // operator-=

// =================================================================
// =================================================================
template<size_t dim>
KOKKOS_INLINE_FUNCTION
StateNd<dim>& operator*(StateNd<dim>& lhs, real_t rhs)
{

  for (size_t i=0; i<dim; ++i)
    lhs[i] *= rhs;

  return lhs;

} // operator+=

} // namespace dyablo

#endif // HYDRO_STATE_H_
