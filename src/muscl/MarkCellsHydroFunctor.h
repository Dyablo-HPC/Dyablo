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

namespace euler_pablo { namespace muscl {

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
                        id2index_t    fm,
                        DataArray Udata,
                        DataArray Udata_ghost) :
    HydroBaseFunctor(params),
    pmesh(pmesh), fm(fm), 
    Udata(Udata), Udata_ghost(Udata_ghost)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
                    DataArray Udata,
		    DataArray Udata_ghost)
  {
    MarkCellsHydroFunctor functor(pmesh, params, fm, 
                                  Udata, Udata_ghost);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

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
                        uint32_t cellId_c,
                        uint32_t cellId_n,
                        bool   isghost_n,
                        real_t dx,
                        real_t pos_c,
                        real_t pos_n) const
  {

    // default returned value
    real_t new_value = current_epsilon;

    return new_value;

  } // update_epsilon

  KOKKOS_INLINE_FUNCTION
  void operator_2d(const size_t i) const 
  {
    constexpr int dim = 2;
    // codim=1 ==> faces
    // codim=2 ==> edges
    const int codim = 1;
    
    // number of faces per cell
    uint8_t nfaces = 2*dim;

    const int nbvar = params.nbvar;

    real_t epsilon;

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

    // current cell center coordinates
    bitpit::darray3 xyz_c = pmesh->getCenter(i);

    // current cell size
    double dx = pmesh->getSize(i);

    // now that we have all neighbors, 
    // we can start computing gradient components

    // sweep neighbors to compute minmod limited gradient
    for (uint16_t j = 0; j < neigh_all.size(); ++j) {
      
      // neighbor index
      uint32_t i_n = neigh_all[j];
      
      // neighbor cell center coordinates
      // if neighbor is a ghost cell, we need to modifiy xyz_c
      bitpit::darray3 xyz_n = pmesh->getCenter(i_n);
      if (isghost_all[j])
        xyz_n = pmesh->getCenterGhost(i_n);
      
      epsilon = update_epsilon(epsilon, i, i_n, isghost_all[j],
                               dx, xyz_c[IX], xyz_n[IX]);  
      
    } // end update epsilon
    
    // mark cell - TODO
    
  } // operator_2d

  KOKKOS_INLINE_FUNCTION
  void operator_3d(const size_t i) const {
  
  } // operator_3d

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t i) const
  {
    
    if (this->params.dimType == TWO_D)
      operator_2d(i);
    
    if (this->params.dimType == THREE_D)
      operator_3d(i);
    
  } // operator ()
  
  std::shared_ptr<AMRmesh> pmesh;
  id2index_t   fm;
  DataArray    Udata;
  DataArray    Udata_ghost;
  
}; // MarkCellsHydroFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // MARK_CELLS_HYDRO_FUNCTOR_H_
