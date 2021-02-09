#pragma once

#include "muscl_block/utils_block.h"
#include "shared/io_utils.h"

namespace dyablo
{

class LightOctree_hashmap;
class LightOctree_pablo;

namespace muscl_block
{
/// @brief Kokkos functor to fill new octants from refinement/coarsening
class MapUserDataFunctor
{
public:  
  /** 
   * @brief Fill new octants created during refinement/coarsening of the octree
   * @param amr_mesh newly refined/coarsened mesh
   * @param Usrc User data before mesh refinement
   * @param Usrc_ghost ghost user data before mesh refinement
   *                   newly coarsened octants might need ghost octant to accumulate smaller 
   *                   octants into new octant
   * @param Udest Used data for newly refined AMR tree. This will be resized inside this function
   **/
  static void apply(  const LightOctree_pablo& lmesh_old,
                      const LightOctree_pablo& lmesh_new,
                      ConfigMap configMap,
                      blockSize_t blockSizes,
                      DataArrayBlock Usrc,
                      DataArrayBlock Usrc_ghost,
                      DataArrayBlock& Udest  );

  static void apply(  const LightOctree_hashmap& lmesh_old,
                      const LightOctree_hashmap& lmesh_new,
                      ConfigMap configMap,
                      blockSize_t blockSizes,
                      DataArrayBlock Usrc,
                      DataArrayBlock Usrc_ghost,
                      DataArrayBlock& Udest  );
};

} // namespace muscl_block
} // namespace dyablo