#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

#include "shared/real_type.h"
#include "shared/utils.h"

namespace dyablo {

using Device = Kokkos::DefaultExecutionSpace;

enum KokkosLayout {
  KOKKOS_LAYOUT_LEFT,
  KOKKOS_LAYOUT_RIGHT
};

/**
 * DataArray mostly used when using one cell per octree leaf.
 *
 * first index is leaf id (curvilinear index along the Morton curve)
 * last index is hydro variable
 */
using DataArray     = Kokkos::View<real_t**, Device>;
using DataArrayHost = DataArray::HostMirror;

/**
 * DataArrayBlock used when designing a solver with a block of
 * data per leaf of the octree.
 *
 * first index identifies a cell inside a block (left layout, from 0 to bx by -1)
 * second index identifies the variable (rho, momentum, energy, ...)
 * third index is the leaf id (curvilinear index along the Morton curve)
 *
 * Note that we enforce Left layout here, since we plan to the Kokkos TeamPolicy with one team per leaf, so we favor memory locality inside a block.
 */
using DataArrayBlock = Kokkos::View<real_t***, Kokkos::LayoutLeft, Device>;
using DataArrayBlockHost = DataArrayBlock::HostMirror;

/**
 * FlagArrayBlock used for flagging faces or block related content, 
 * this is a 1D array (along space filling curve)).
 */
using FlagArrayBlock = Kokkos::View<uint16_t*, Device>;
using FlagArrayBlockHost = FlagArrayBlock::HostMirror;

// =============================================================
// =============================================================
/**
 * a dummy swap device routine.
 */
template <class T>
KOKKOS_INLINE_FUNCTION void my_swap(T& a, T& b) {
  T c{std::move(a)};
  a = std::move(b);
  b = std::move(c);
} // my_swap

} // namespace dyablo

#endif // KOKKOS_SHARED_H_
