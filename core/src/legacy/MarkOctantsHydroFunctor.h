#pragma once 

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "HydroState.h"

#include "amr/LightOctree.h"

// utils hydro
#include "utils_hydro.h"

#include "utils_block.h"

namespace dyablo { 
namespace muscl_block {

/*************************************************/
/**
 * Mark octants for refine or coarsen according to a
 * gradient-like conditions.
 *
 * We adopt here a different strategy from muscl, here
 * in muscl_block we have one block of cells per leaf (i.e. octant).
 * 
 * To mark an octant for refinement, we have adapted ideas 
 * from code AMUN, i.e. for each inner cells, we apply the
 * refinement criterium, then reduce the results to have a
 * single indicator to decide whether or not the octant is
 * flag for refinement.
 *
 * We do not use directly the global array U (of conservative variables
 * in block, without ghost cells) but Qgroup containing a smaller 
 * number of octants, but with ghost cells. This means the computation
 * is organized in a piecewise fashion, i.e. this functor is aimed
 * to be called inside a loop over all sub-groups of octants.
 *
 */
class MarkOctantsHydroFunctor {

public:
  /**
   * Structure ton contain refine/coarsen markers
   * 
   * Markers are stored in a sparse 'coordinate list' storage to reduce 
   * the amount of data to transfer from GPU to CPU
   **/
  struct markers_t {
    Kokkos::View<uint32_t*> iOcts;
    Kokkos::View<int*> markers;
    Kokkos::View<uint32_t> count;

    /// Marker list tha can store at most 'capacity' elements
    markers_t(uint32_t capacity);
    /**
     * Append a (iOct, marker) pair to the list 
     * does not reallocate storage : do not exceed capacity
     **/
    KOKKOS_INLINE_FUNCTION void push_back(uint32_t iOct, int marker) const;
    /// Get number of markers [Host Only]
    uint32_t size();
    /// Get Host View for octant index
    Kokkos::View<uint32_t*>::HostMirror getiOcts_host();
    /// Get Host View markers associated with octant index from getiOcts_host
    Kokkos::View<int*>::HostMirror getMarkers_host();
  };


public:  
  /**
   * Mark cells for refine/coarsen functor.
   *
   * \param[in] lmesh AMR mesh
   * \param[in] fm field map to access user data
   * \param[in] Qgroup primitive variables (block data with ghost cells)
   * \param[in] iGroup identifies a group of octants among all subgroup
   * \param[in] epsilon_refine threshold value
   * \param[in] epsilon_coarsen threshold value
   * \param[inout] markers used to refine/coarsen (see markers_t)
   **/
  static void apply(LightOctree    lmesh,
                    int level_min, int level_max,
		                id2index_t     fm,
                    blockSize_t    blockSizes,
                    uint32_t       ghostWidth,
                    uint32_t       nbOcts,
                    uint32_t       nbOctsPerGroup,
                    DataArrayBlock Qgroup,
                    uint32_t       iGroup,
                    real_t         error_min,
                    real_t         error_max,
                    markers_t      markers );
  /// Write markers to PABLO mesh
  static void set_markers_pablo(markers_t markers, AMRmesh& pmesh);

}; // class MarkOctantsHydroFunctor

} // namespace muscl_block
} // namespace dyablo
