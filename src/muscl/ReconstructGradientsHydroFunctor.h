/**
 * \file ReconstructGradientsHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef RECONSTRUCT_GRADIENTS_HYDRO_FUNCTOR_H_
#define RECONSTRUCT_GRADIENTS_HYDRO_FUNCTOR_H_

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
 * Reconstruct gradients functor.
 *
 * Equivalent to slope limiter in a regular cartesian grid code.
 *
 */
class ReconstructGradientsHydroFunctor : public HydroBaseFunctor {
  
public:
  /**
   * Reconstruct gradients functor constructor.
   *
   * \param[in] pmesh AMRmesh Pablo data structure
   * \param[in] params
   * \param[in] fm field map to access user data
   * \param[in] Qdata primitive variables
   * \param[in] Qdata_ghost primitive variables in ghost cells (only meaningful when MPI activated)
   * \param[out] SlopeX limited slopes along x data array
   * \param[out] SlopeY limited slopes along y data array
   * \param[out] SlopeZ limited slopes along z data array
   *
   */
  ReconstructGradientsHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
				   HydroParams params,
				   id2index_t    fm,
				   DataArray Qdata,
				   DataArray Qdata_ghost,
				   DataArray SlopeX,
				   DataArray SlopeY,
				   DataArray SlopeZ) :
    HydroBaseFunctor(params),
    pmesh(pmesh), fm(fm), 
    Qdata(Qdata), Qdata_ghost(Qdata_ghost),
    SlopeX(SlopeX), SlopeY(SlopeY), SlopeZ(SlopeZ)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
                    DataArray Qdata,
		    DataArray Qdata_ghost,
		    DataArray SlopeX,
		    DataArray SlopeY,
		    DataArray SlopeZ)
  {
    ReconstructGradientsHydroFunctor functor(pmesh, params, fm, 
                                             Qdata, Qdata_ghost,
                                             SlopeX, SlopeY, SlopeZ);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  /**
   * Update limited gradient with information from a given neighbor.
   *
   * \param[in] current_gradient is the current value (to be updated)
   * \param[in] cellId_c is current cell id
   * \param[in] cellId_n is neighbor cell id
   * \param[in] isghost_n boolean specifying is neighbor is a ghost
   * \param[in] dx is current cell size
   * \param[in] pos_c is current  cell coordinates (either x,y or z)
   * \param[in] pos_n is neighbor cell coordinates (either x,y or z)
   * \param[in] ivar identifies the working (primitive) variable
   * \param[in] dir identifies which component of the gradient
   * \return updated limiter gradient
   */
  KOKKOS_INLINE_FUNCTION
  real_t update_minmod(real_t current_grad,
                       uint32_t cellId_c,
                       uint32_t cellId_n,
                       bool   isghost_n,
                       real_t dx,
                       real_t pos_c,
                       real_t pos_n,
                       int ivar,
                       int dir) const
  {

    // default returned value (limited gradient didn't change)
    real_t new_value = current_grad;

    // compute distance along direction "dir" between current and
    // neighbor cell
    real_t delta_x = fabs(pos_n - pos_c);

    // only check if we truly have a neighbor along given dir
    if (delta_x > 0.5 * dx) {

      real_t new_grad;

      // is neighbor a ghost ?
      if (isghost_n) {
      
        // left or right neighbor ?
        new_grad = pos_n > pos_c ?
          Qdata_ghost(cellId_n, fm[ivar]) - Qdata(cellId_c, fm[ivar]) :
          Qdata(cellId_c, fm[ivar])       - Qdata_ghost(cellId_n, fm[ivar]);
        new_grad /= delta_x;
        
      } else {
        
        // left or right neighbor ?
        new_grad =
            pos_n > pos_c
                ? Qdata(cellId_n, fm[ivar]) - Qdata(cellId_c, fm[ivar])
                : Qdata(cellId_c, fm[ivar]) - Qdata(cellId_n, fm[ivar]);
        new_grad /= delta_x;

      }

      /*
       * this is minmod: limited gradient may need to be updated
       */

       // this first test ensure a correct initialization
#ifdef __CUDA_ARCH__
      if ( current_grad == CUDART_INF )
        new_value = new_grad;
#else
      if ( current_grad == std::numeric_limits<real_t>::max() )
        new_value = new_grad;
#endif
      else if (current_grad * new_grad < 0)
        new_value = 0.0;
      else if ( fabs(new_grad) < fabs(current_grad) )
        new_value = new_grad;
    }

    return new_value;

  } // update_minmod

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

    // temp variables for gradient
    Kokkos::Array<real_t,dim> grad;

    grad[IX] = 0;
    grad[IY] = 0;

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

    // for each (primitive) variable, compute limited gradient
    for (uint8_t ivar = 0; ivar<nbvar; ++ivar) {

      // initialize gradient components with something very large, since
      // we are doing a minmod slope limiter, as soon as a genuine 
      // neighbor is found, gradient components will be updated to
      // something reasonable
      // watch out the sign of the slope might be wrong here, 
      // but it will later be corrected inside update_minmod
#ifdef __CUDA_ARCH__
      grad[IX] = CUDART_INF;
      grad[IY] = CUDART_INF;
#else
      grad[IX] = std::numeric_limits<real_t>::max();
      grad[IY] = std::numeric_limits<real_t>::max();
#endif // __CUDA_ARCH__

      // sweep neighbors to compute minmod limited gradient
      for (uint16_t j = 0; j < neigh_all.size(); ++j) {

        // neighbor index
        uint32_t i_n = neigh_all[j];

        // neighbor cell center coordinates
        bitpit::darray3 xyz_n = pmesh->getCenter(i_n);

        grad[IX] = update_minmod(grad[IX], i, i_n, isghost_all[j],
                                 dx, xyz_c[IX], xyz_n[IX],
                                 ivar, IX);

        grad[IY] = update_minmod(grad[IY], i, i_n, isghost_all[j],
                                 dx, xyz_c[IY], xyz_n[IY],
                                 ivar, IY);

      } // end minmod

      // copy back limited gradient
      SlopeX(i,fm[ivar]) = grad[IX];
      SlopeY(i,fm[ivar]) = grad[IY];

    } // end for ivar
    
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
  DataArray    Qdata;
  DataArray    Qdata_ghost;
  DataArray    SlopeX, SlopeY, SlopeZ;
  
}; // ReconstructGradientsHydroFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // RECONSTRUCT_GRADIENTS_HYDRO_FUNCTOR_H_
