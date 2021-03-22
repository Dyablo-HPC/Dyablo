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

// utils hydro
#include "shared/utils_hydro.h"

namespace dyablo { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Mark cells for refine or coarsen according to a gradient-like conditions.
 *
 */
class MarkCellsHydroFunctor {
  
public:
  /**
   * Mark cells for refine/coarsen functor.
   *
   * \param[in] lmesh LightOctree data structure
   * \param[in] params
   * \param[in] fm field map to access user data
   * \param[in] Udata conservative variables
   * \param[in] Udata_ghost conservative variables in ghost cells (only meaningful when MPI activated)
   * \param[in] eps_refine is a threshold, we do refine when criterium is larger than this value
   * \param[in] eps_coarsen is a threshold, we do unrefine when criterium is smaller than this value
   *
   */
  MarkCellsHydroFunctor(LightOctree lmesh,
                        HydroParams params,
                        id2index_t  fm,
                        DataArray   Udata,
                        DataArray   Udata_ghost,
                        Kokkos::View<int8_t*> marker,
                        real_t      eps_refine,
                        real_t      eps_coarsen) :
    lmesh(lmesh),
    params(params),
    fm(fm),
    Udata(Udata), 
    Udata_ghost(Udata_ghost),
    marker(marker),
    epsilon_refine(eps_refine),
    epsilon_coarsen(eps_coarsen)
  {};
  
  //! static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
        LightOctree lmesh,
		    HydroParams params,
		    id2index_t  fm,
                    DataArray   Udata,
		    DataArray   Udata_ghost,
                    real_t      eps_refine,
                    real_t      eps_coarsen)
  {
    uint32_t nbOct = lmesh.getNumOctants();

    Kokkos::View<int8_t*> marker("MarkCellsHydroFunctor markers", nbOct);
    MarkCellsHydroFunctor functor(lmesh, params, fm, 
                                  Udata, Udata_ghost,
                                  marker,
                                  eps_refine, 
                                  eps_coarsen);
    Kokkos::parallel_for("dyablo::muscl::MarkCellsHydroFunctor", nbOct, functor);

    auto marker_host = Kokkos::create_mirror_view(marker);
    Kokkos::deep_copy(marker_host, marker);

    for( uint32_t i=0; i<nbOct; i++ )
    {
      pmesh->setMarker(i, marker_host(i));
    }
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
   * \param[in] current_epsilon is the current value (to be updated)
   * \param[in] i is current cell id
   * \param[in] i_n is neighbor cell id
   * \param[in] isghost_n boolean specifying is neighbor is a ghost
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

  template<uint8_t dir>
  KOKKOS_INLINE_FUNCTION
  bool face_along_axis(uint8_t iface) const
  {
    return ( iface>>1 == dir );

  } // face_along_axis

  KOKKOS_INLINE_FUNCTION
  void operator()(const uint32_t i) const
  {
    const int dim = this->params.dimType == TWO_D ? 2 : 3;
    
    // number of faces per cell
    uint8_t nfaces = 2*dim;

    real_t epsilon = 0.0;

    for (uint8_t iface=0; iface<nfaces; ++iface) 
    {

      // Generate neighbor relative position from iface
      LightOctree::offset_t offset = {0};
      if( face_along_axis<IX>(iface) ) offset[IX] = (iface & 0x1) == 0 ? -1 : 1;
      if( face_along_axis<IY>(iface) ) offset[IY] = (iface & 0x1) == 0 ? -1 : 1;
      if( face_along_axis<IZ>(iface) ) offset[IZ] = (iface & 0x1) == 0 ? -1 : 1;

      if (lmesh.isBoundary({i, false}, offset))
        continue;

      LightOctree::NeighborList neighbors = lmesh.findNeighbors( {i, false}, offset );

      for( int j=0; j<neighbors.size(); j++ )
      {
        epsilon = update_epsilon(epsilon, i, neighbors[j].iOct, neighbors[j].isGhost);
      }
    }
    
    // now epsilon has been computed, we can mark / flag cells for 
    // refinement or coarsening

    // get current cell level
    uint8_t level = lmesh.getLevel({i,false});

    // epsilon too large, set octant to be refined
    if ( level < params.level_max and epsilon > epsilon_refine )
      marker(i) = 1;
    // epsilon too small, set octant to be coarsened
    else if ( level > params.level_min and epsilon < epsilon_coarsen)
      marker(i) = -1;
    else 
      marker(i) = 0;
    
  } // operator ()
  
  LightOctree lmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  DataArray    Udata_ghost;
  Kokkos::View<int8_t*> marker;
  real_t       epsilon_refine;
  real_t       epsilon_coarsen;

}; // MarkCellsHydroFunctor

} // namespace muscl

} // namespace dyablo

#endif // MARK_CELLS_HYDRO_FUNCTOR_H_
