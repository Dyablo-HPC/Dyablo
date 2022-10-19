#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>

#include "real_type.h"
#include "utils/misc/utils.h"

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
using DataArray     = Kokkos::View<real_t**, Kokkos::LayoutLeft, Device>;
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


} // namespace dyablo

#endif // KOKKOS_SHARED_H_
