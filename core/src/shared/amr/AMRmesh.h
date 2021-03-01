#pragma once

#include "AMRmesh_pablo.h"
#include "AMRmesh_hashmap.h"

#define DYABLO_USE_GPU_MESH

namespace dyablo{
#ifdef DYABLO_USE_GPU_MESH
using AMRmesh = AMRmesh_hashmap;
#else
using AMRmesh = AMRmesh_pablo;
#endif
}