#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

#include "shared/real_type.h"
#include "shared/utils.h"

using Device = Kokkos::DefaultExecutionSpace;

enum KokkosLayout {
  KOKKOS_LAYOUT_LEFT,
  KOKKOS_LAYOUT_RIGHT
};

// last index is hydro variable
// n-1 first indexes are space (i,j,k,....)
typedef Kokkos::View<real_t**, Device> DataArray;
typedef DataArray::HostMirror          DataArrayHost;

#endif // KOKKOS_SHARED_H_
