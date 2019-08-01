/**
 * \file MarkCellsHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef MARK_CELLS_HYDRO_FUNCTOR_H_
#define MARK_CELLS_HYDRO_FUNCTOR_H_

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// base class
#include "muscl/HydroBaseFunctor.h"

namespace dyablo { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Mark cells for refine or coarsen according to a gradient-like conditions.
 *
 */
class MarkCellsHydroFunctor : public HydroBaseFunctor {
  
public:
  /**
   * Mark cells for refine/coarsen functor.
   *
   * \param[in] pmesh AMRmesh Pablo data structure
   * \param[in] params
   * \param[in] fm field map to access user data
   * \param[in] Udata conservative variables
   * \param[in] Udata_ghost conservative variables in ghost cells (only meaningful when MPI activated)
   *
   */
  MarkCellsHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
                        HydroParams params,
                        id2index_t  fm,
                        DataArray   Udata,
                        DataArray   Udata_ghost,
                        real_t      eps_refine,
                        real_t      eps_coarsen) :
    HydroBaseFunctor(params),
    pmesh(pmesh),
    fm(fm),
    Udata(Udata), 
    Udata_ghost(Udata_ghost),
    epsilon_refine(eps_refine),
    epsilon_coarsen(eps_coarsen)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
                    DataArray   Udata,
		    DataArray   Udata_ghost,
                    real_t      eps_refine,
                    real_t      eps_coarsen)
  {
    MarkCellsHydroFunctor functor(pmesh, params, fm, 
                                  Udata, Udata_ghost,
                                  eps_refine, 
                                  eps_coarsen);
    Kokkos::parallel_for("dyablo::muscl::MarkCellsHydroFunctor", pmesh->getNumOctants(), functor);
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

    if (isghost_n) {
      new_value = indicator_scalar_gradient (Udata(i,fm[ID]), 
                                             Udata_ghost(i_n,fm[ID]));
    } else {
      new_value = indicator_scalar_gradient (Udata(i,fm[ID]), 
                                             Udata(i_n,fm[ID]));    
    }

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

    // this vector contains quad ids
    // corresponding to neighbors
    std::vector<uint32_t> neigh; // through a given face
    std::vector<uint32_t> neigh_all; // all neighbors

    // this vector contains ghost status of each neighbors
    std::vector<bool> isghost; // through a given face
    std::vector<bool> isghost_all; // all neighbors

    // only sweep neighbors through faces
    for (uint8_t iface=0; iface<nfaces; ++iface) {
      pmesh->findNeighbours(i,iface,codim,neigh,isghost);
      
      // insert data into all neighbor lists
      neigh_all.insert(neigh_all.end(), neigh.begin(), neigh.end());
      isghost_all.insert(isghost_all.end(), isghost.begin(), isghost.end());
    }

    // now that we have all neighbors, 
    // we can start apply refinement criterium

    // sweep neighbors to compute minmod limited gradient
    for (uint16_t j = 0; j < neigh_all.size(); ++j) {
      
      // neighbor index
      uint32_t i_n = neigh_all[j];
            
      epsilon = update_epsilon(epsilon, i, i_n, isghost_all[j]);  
      
    } // end update epsilon
    
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
  id2index_t   fm;
  DataArray    Udata;
  DataArray    Udata_ghost;
  real_t       epsilon_refine;
  real_t       epsilon_coarsen;

}; // MarkCellsHydroFunctor

} // namespace muscl

} // namespace dyablo

#endif // MARK_CELLS_HYDRO_FUNCTOR_H_
