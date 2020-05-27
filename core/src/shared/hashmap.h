#ifndef OCTREE_HASHMAP_H
#define OCTREE_HASHMAP_H

#include <kokkos_shared.h>

namespace dyablo {

using hashkey_t = uint64_t;

/**
 * hashmap_t is a hash map where
 * - key is the global Morton index
 * - value is a uint32_t (the local index to an octant leaf of a PABLO 
 * octree). Here we assume that for most application 32 bits is enought
 * to address the whole number of octant per MPI process.
 *
 * An instance of hashmap_t can be used to address both:
 * - metadata array
 * - heavy (payload) data which is application specific (rho, momentum, ...)
 */
using hashmap_t = Kokkos::Unordered_Map<hashkey_t, uint32_t>;

// TODO - create a functor on host to initialize the hashmap using a
// PABLO uniform instance as input.

} // namespace dyablo

#endif // OCTREE_HASHMAP_H
