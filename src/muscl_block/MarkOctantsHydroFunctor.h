// !!!!!!!!!!!!!!!!!!!!!!!!!!
// UNFINISHED
// !!!!!!!!!!!!!!!!!!!!!!!!!!
/**
 * \file MarkOctantsHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef DYABLO_MUSCL_BLOCK_MARK_OCTANTS_HYDRO_FUNCTOR_H_
#define DYABLO_MUSCL_BLOCK_MARK_OCTANTS_HYDRO_FUNCTOR_H_

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// utils hydro
#include "shared/utils_hydro.h"

namespace dyablo { namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Mark octants for refine or coarsen according to a
 * gradient-like conditions.
 *
 * We adopt here a different strategy from muscl, here
 * in muscl we have one block of cells per leaf (i.e. octant).
 * 
 * To mark an octant for refinement, we have adapted ideas 
 * from code AMUN, i.e. for each inner cells, we apply the
 * refinement criterium, then reduce the results to have a
 * single indicator to decide whether or not the octant is
 * flag for refinement.
 *
 * We do not use directly the global array U (of conservative variables
 * in block, without ghost cells) but Ugroup containing a smaller 
 * number of octants, but with ghost cells. This means the computation
 * is organized in a piecewise fashion, i.e. this functor is aimed
 * to be called inside a loop over all sub-groups of octants.
 *
 */
class MarkOctantsHydroFunctor {
  
public:
  /**
   * Mark cells for refine/coarsen functor.
   *
   * \param[in] pmesh AMR mesh Pablo data structure
   * \param[in] params
   * \param[in] fm field map to access user data
   * \param[in] Ugroup conservative variables (block data with ghost cells)
   * \param[in] iGroup identifies a group of octants among all subgroup
   * \param[in] epsilon_refine threshold value
   * \param[in] epsilon_coarsen threshold value
   *
   *
   * \todo refactor interface : there is no need to have a PabloUniform (pmesh)
   * object here; it is only used to call setMarker. All we need is to make
   * Pablo exposes the array of refinement flags (as Kokkos::View).
   *
   * \todo the total number of octants (for current MPI process) is retrieve from
   * pmesh object; should be passed as an argument.
   *
   */
  MarkOctantsHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
                          HydroParams    params,
                          id2index_t     fm,
                          DataArrayBlock Ugroup,
                          uint32_t       iGroup,
                          real_t         eps_refine,
                          real_t         eps_coarsen) :
    pmesh(pmesh),
    params(params),
    fm(fm),
    Ugroup(Ugroup), 
    iGroup(iGroup), 
    epsilon_refine(eps_refine),
    epsilon_coarsen(eps_coarsen)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams    params,
		    id2index_t     fm,
                    DataArrayBlock Ugroup,
                    uint32_t       iGroup,
                    real_t         eps_refine,
                    real_t         eps_coarsen)
  {
    MarkOctantsHydroFunctor functor(pmesh, params, fm, 
                                    Ugroup, iGroup,
                                    eps_refine, 
                                    eps_coarsen);

    // todo : change range policy into team policy
    Kokkos::parallel_for("dyablo::muscl::MarkOctantsHydroFunctor",
                         pmesh->getNumOctants(), 
                         functor);
  } // apply

  /**
   * Indicator - scalar gradient.
   *
   * Adapted from CanoP for comparison.
   * returned value is between 0 and 1.
   * - a small value probably means no refinement necessary
   * - a high value probably means refinement should be activated
   */
  KOKKOS_INLINE_FUNCTION
  real_t
  indicator_scalar_gradient (real_t qi, real_t qj) const
  {
    
    real_t max = fmax (fabs (qi), fabs (qj));
    
    if (max < 0.001) {
      return 0;
    }
    
    max = fabs (qi - qj) / max;
    return fmax (fmin (max, 1.0), 0.0);
    
  } // indicator_scalar_gradient

  /**
   * Update epsilon with information from a given neighbor.
   *
   * \param[in] current_gradient is the current value (to be updated)
   * \param[in] cellId_c is current cell id
   * \param[in] cellId_n is neighbor cell id
   * \param[in] isghost_n boolean specifying is neighbor is a ghost
   * \param[in] dx is current cell size
   * \param[in] pos_c is current  cell coordinates (either x,y or z)
   * \param[in] pos_n is neighbor cell coordinates (either x,y or z)
   * \return updated limiter gradient
   */
  KOKKOS_INLINE_FUNCTION
  real_t update_epsilon(real_t current_epsilon,
                        uint32_t i,
                        uint32_t i_n,
                        bool   isghost_n) const
  {

    // default returned value
    real_t new_value = current_epsilon;

    // if (isghost_n) {
    //   new_value = indicator_scalar_gradient (Udata(i,fm[ID]), 
    //                                          Udata_ghost(i_n,fm[ID]));
    // } else {
    //   new_value = indicator_scalar_gradient (Udata(i,fm[ID]), 
    //                                          Udata(i_n,fm[ID]));    
    // }

    return fmax(new_value,current_epsilon);

  } // update_epsilon

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const
  {
    
    const int dim = this->params.dimType == TWO_D ? 2 : 3;
    
    // codim=1 ==> faces
    // codim=2 ==> edges
    const int codim = 1;
    
    // number of faces per cell
    uint8_t nfaces = 2*dim;

    real_t epsilon = 0.0;
    
    // now epsilon has been computed, we can mark / flag cells for 
    // refinement or coarsening

    // get current cell level
    uint8_t level = pmesh->getLevel(i);

    // epsilon too large, set octant to be refined
    if ( level < params.level_max and epsilon > epsilon_refine )
      pmesh->setMarker(i,1);

    // epsilon too small, set octant to be coarsened
    if ( level > params.level_min and epsilon < epsilon_coarsen)
      pmesh->setMarker(i,-1);
    
  } // operator ()
  
  std::shared_ptr<AMRmesh> pmesh;
  HydroParams    params;
  id2index_t     fm;
  DataArrayBlock Ugroup;
  uint32_t       iGroup;
  real_t         epsilon_refine;
  real_t         epsilon_coarsen;

}; // MarkOctantsHydroFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // DYABLO_MUSCL_BLOCK_MARK_OCTANTS_HYDRO_FUNCTOR_H_
