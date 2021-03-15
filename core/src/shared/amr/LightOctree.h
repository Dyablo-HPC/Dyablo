#pragma once

#include "shared/amr/LightOctree_hashmap.h"
#include "shared/amr/LightOctree_pablo.h"

namespace dyablo { 

#ifdef KOKKOS_ENABLE_CUDA
using LightOctree = LightOctree_hashmap;
#else
using LightOctree = LightOctree_pablo;
#endif

} //namespace dyablo
