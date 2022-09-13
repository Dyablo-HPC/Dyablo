#pragma once

namespace dyablo { 

// class LightOctree_hashmap;
// class LightOctree_pablo;

// #ifdef KOKKOS_ENABLE_CUDA
// using LightOctree = LightOctree_hashmap;
// #else
// using LightOctree = LightOctree_pablo;
// #endif

class LightOctree_hashmap;
using LightOctree = LightOctree_hashmap;

} //namespace dyablo
