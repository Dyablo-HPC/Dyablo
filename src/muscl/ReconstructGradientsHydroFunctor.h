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
   * Reconstruct gradients
   *
   * \param[in] params
   * \param[in] Udata conservative variables - needed ???
   * \param[in] Qdata primitive variables
   */
  ReconstructGradientsHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
				   HydroParams params,
				   id2index_t    fm,
				   DataArray Udata,
				   DataArray Qdata,
				   DataArray SlopeX,
				   DataArray SlopeY,
				   DataArray SlopeZ) :
    HydroBaseFunctor(params),
    pmesh(pmesh), fm(fm), Udata(Udata), Qdata(Qdata),
    SlopeX(SlopeX), SlopeY(SlopeY), SlopeZ(SlopeZ)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
		    DataArray Udata,
                    DataArray Qdata,
		    DataArray SlopeX,
		    DataArray SlopeY,
		    DataArray SlopeZ)
  {
    ReconstructGradientsHydroFunctor functor(pmesh, params, fm, Udata, Qdata,SlopeX,SlopeY,SlopeZ);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  /**
   * Update limited gradient with information from a given neighbor.
   *
   * \param[in] current_gradient is the current value (to be updated)
   * \param[in] cellId_c is current cell id
   * \param[in] cellId_n is neighbor cell id
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

      // left or right neighbor ?
      real_t new_grad = pos_n - pos_c > 0 ?
        Udata(cellId_n,fm[ivar]) - Udata(cellId_c,fm[ivar]) :
        Udata(cellId_c,fm[ivar]) - Udata(cellId_n,fm[ivar]) ;
      new_grad /= delta_x;
      
      // this is minmod: limited gradient might need to be update
      if (current_grad * new_grad < 0)
        new_value = 0.0;
      else if ( fabs(new_grad) < fabs(current_grad) )
        new_value = new_grad;

    }

    return new_value;

  } // update_minmod

  KOKKOS_INLINE_FUNCTION
  void operator_2d(const size_t &i) const 
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

      // initialize gradient
      Kokkos::Array<bool,dim> grad_ok;
      grad_ok[IX] = false;
      grad_ok[IY] = false;
      
      // init gradx, grady
      for (uint16_t j = 0; j < neigh.size(); j++) {

        // neighbor index
        uint32_t i_n = neigh_all[j];

        // neighbor cell center coordinates
        bitpit::darray3 xyz_n = pmesh->getCenter(i_n);

        // distance between current and neighbor cell
        real_t delta_x, delta_y;

        /*
         * check if we found a neigh along x
         */

        // start by evaluating delta x
        delta_x = fabs(xyz_n[IX] - xyz_c[IX]);
        
        // correct delta_x if needed (when i and i_n are actually
        // accross external border with periodic boundaries)
        if (delta_x>2*dx)
          delta_x = (pmesh->getSize(i)+pmesh->getSize(i_n))/2;

        if (grad_ok[IX]==false and 
            ( delta_x > 0.5 * dx) ) {
          grad_ok[IX] = true;
          grad[IX] = xyz_n[IX] - xyz_c[IX] > 0.5 * dx ? 
            Udata(i_n,fm[ivar]) - Udata(i  ,fm[ivar]) :
            Udata(i  ,fm[ivar]) - Udata(i_n,fm[ivar]) ;
          grad[IX] /= delta_x;
        }

        /*
         * check if we found a neigh along y
         */

        // start by evaluating delta x
        delta_y = fabs(xyz_n[IY] - xyz_c[IY]);

        // correct delta_y if needed (when i and i_n are actually
        // accross external border with periodic boundaries)
        if (delta_y>2*dx) 
          delta_y = (pmesh->getSize(i)+pmesh->getSize(i_n))/2;
        
        if (grad_ok[IY]==false and 
            (delta_y > 0.5 * dx) ) {
          grad_ok[IY] = true;
          grad[IY] = xyz_n[IY] - xyz_c[IY] > 0.5 * dx ? 
            Udata(i_n,fm[ivar]) - Udata(i  ,fm[ivar]) :
            Udata(i  ,fm[ivar]) - Udata(i_n,fm[ivar]) ;
          grad[IY] /= delta_y;
        }

        if (grad_ok[IX] and grad_ok[IY])
          break; // initialization done

      } // end initialize gradx, grady

      //if (ivar==ID) printf("kkk2 %d %f || %f %f || %f %f\n",ivar, Udata(i, fm[ivar]), grad[IX],grad[IY],xyz_c[IX],xyz_c[IY]);

      // sweep neighbors to compute minmod limited gradient
      for (uint16_t j = 0; j < neigh.size(); j++) {

        // neighbor index
        uint32_t i_n = neigh_all[j];

        // neighbor cell center coordinates
        bitpit::darray3 xyz_n = pmesh->getCenter(i_n);

        grad[IX] = update_minmod(grad[IX], i, i_n,
                                 dx, xyz_c[IX], xyz_n[IX],
                                 ivar, IX);

        grad[IY] = update_minmod(grad[IY], i, i_n,
                                 dx, xyz_c[IY], xyz_n[IY],
                                 ivar, IY);

      } // end minmod

      //if (ivar==ID) printf("kkk2 %d %f || %f %f || %f %f\n",ivar, Udata(i, fm[ivar]), grad[IX],grad[IY],xyz_c[IX],xyz_c[IY]);
      // copy back limited gradient
      SlopeX(i,fm[ivar]) = grad[IX];
      SlopeY(i,fm[ivar]) = grad[IY];

    } // end for ivar
    
    
  } // operator_2d

  KOKKOS_INLINE_FUNCTION
  void operator_3d(const size_t &i) const {
  
  } // operator_3d

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {
    
    if (this->params.dimType == TWO_D)
      operator_2d(i);
    
    if (this->params.dimType == THREE_D)
      operator_3d(i);
    
  } // operator ()
  
  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  id2index_t   fm;
  DataArray    Udata;
  DataArray    Qdata;
  DataArray    SlopeX, SlopeY, SlopeZ;
  
}; // ReconstructGradientsHydroFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // RECONSTRUCT_GRADIENTS_HYDRO_FUNCTOR_H_
