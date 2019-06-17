/**
 * \file ComputeFluxesAndUpdateHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef COMPUTE_FLUXES_AND_UPDATE_HYDRO_FUNCTOR_H_
#define COMPUTE_FLUXES_AND_UPDATE_HYDRO_FUNCTOR_H_

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
 * Compute Fluxes and Update functor.
 *
 * Loop through all cell (sub-)faces:
 * - compute fluxes: need first to reconstruct data on both sides
 *   of an interface, then call a Riemann solver
 * - update only current cell
 *
 */
class ComputeFluxesAndUpdateHydroFunctor : public HydroBaseFunctor {

private:
  using offsets_t = Kokkos::Array<real_t,3>;

public:
  /**
   * Reconstruct gradients
   *
   * \param[in]  pmesh pointer to AMR mesh structure
   * \param[in]  params
   * \param[in]  fm field map
   * \param[in]  Data_in current time step data (conservative variables)
   * \param[out] Data_out next time step data (conservative variables)
   * \param[in]  Qdata primitive variables
   * \param[in]  Slopes_x limited slopes along x axis
   * \param[in]  Slopes_y limited slopes along y axis
   * \param[in]  Slopes_z limited slopes along z axis
   * \
   */
  ComputeFluxesAndUpdateHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
                                     HydroParams params,
                                     id2index_t    fm,
                                     DataArray Data_in,
                                     DataArray Data_out,
                                     DataArray Qdata,
                                     DataArray SlopeX,
                                     DataArray SlopeY,
                                     DataArray SlopeZ,
                                     real_t    dt) :
    HydroBaseFunctor(params),
    pmesh(pmesh),
    fm(fm),
    Data_in(Data_in),
    Data_out(Data_out),
    Qdata(Qdata),
    SlopeX(SlopeX),
    SlopeY(SlopeY),
    SlopeZ(SlopeZ),
    dt(dt)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
		    DataArray Data_in,
		    DataArray Data_out,
                    DataArray Qdata,
		    DataArray SlopeX,
		    DataArray SlopeY,
		    DataArray SlopeZ,
                    real_t    dt)
  {
    ComputeFluxesAndUpdateHydroFunctor functor(pmesh, params, fm, 
                                               Data_in, Data_out,
                                               Qdata,
                                               SlopeX,SlopeY,SlopeZ,
                                               dt);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  // =======================================================================
  // =======================================================================
  template<uint8_t dir>
  KOKKOS_INLINE_FUNCTION
  bool face_along_axis(uint8_t iface) const
  {
    return ( (iface>>(dir+1)) == 0 );

  } // face_along_axis

  // =======================================================================
  // =======================================================================
  /**
   * returns offsets in units of delta_x/2.
   *
   * on current cell border where the primitive variables must be reconstructed
   * using the limited slopes.
   *
   * In 2D, offsets lies in the following square (mapping current cell)
   *
   *  (-1,1) --- (0,1) ---- (1,1) 
   *    |          |          |
   *    |          |          |
   *    |          |          |
   *  (-1,0) --- (0,0) ---- (1,0)
   *    |          |          |
   *    |          |          |
   *    |          |          |
   *  (-1,-1) ---(0,-1) --- (1,-1) 
   * 
   *
   * \param[in] i   current  cell id
   * \param[in] i_n neighbor cell id
   * \param[in] iface face id (from current cell side)
   */
  KOKKOS_INLINE_FUNCTION
  offsets_t get_reconstruct_offsets_current_2d(const uint32_t i, 
                                               const uint32_t i_n,
                                               const uint8_t iface) const
  {

    offsets_t offsets;

    uint8_t ifaceX = iface>>IX;
    uint8_t ifaceY = iface>>IY;

    /*
     * - current cell and neighbor cell have the same size
     * or
     * - current cell is smaller than neighbor
     */
    if (pmesh->getSize(i) <= pmesh->getSize(i_n)) {

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 2.0 * ifaceX - 1;
        offsets[IY] = 0.0;
        offsets[IZ] = 0.0;
      }

      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = 0.0;
        offsets[IY] = 2.0 * ifaceY - 1;
        offsets[IZ] = 0.0;
      }

    } // end same size

    /*
     * current cell is larger than neighbors
     */
    if (pmesh->getSize(i) > pmesh->getSize(i_n)) {

      bitpit::darray3 xyz_c = pmesh->getCenter(i);
      bitpit::darray3 xyz_n = pmesh->getCenter(i_n);

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 2.0*ifaceX-1;
        offsets[IY] = xyz_n[IY]>xyz_c[IY] ? -0.5 : 0.5;
        offsets[IZ] = 0.0;
      }
      
      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = xyz_n[IX]>xyz_c[IX] ? 0.5 : -0.5;
        offsets[IY] = 2.0*ifaceY-1;
        offsets[IZ] = 0.0;
      }      

    } // end current cell is larger

    return offsets;

  } // get_reconstruct_offsets_current_2d

  // =======================================================================
  // =======================================================================
  /**
   * returns offsets in units of delta_x/2 (delta_x of neighbor cell).
   *
   * on current cell border where the primitive variables must be reconstructed
   * using the limited slopes.
   *
   * In 2D, offsets lies in the following square (mapping current cell)
   *
   *  (-1,1) --- (0,1) ---- (1,1) 
   *    |          |          |
   *    |          |          |
   *    |          |          |
   *  (-1,0) --- (0,0) ---- (1,0)
   *    |          |          |
   *    |          |          |
   *    |          |          |
   *  (-1,-1) ---(0,-1) --- (1,-1) 
   * 
   *
   * \param[in] i   current  cell id
   * \param[in] i_n neighbor cell id
   * \param[in] iface face id (from current cell side)
   *
   * We use the symetry 2*ifaceX-1 becomes 1-2*ifaceX (same for other direction)
   */
  KOKKOS_INLINE_FUNCTION
  offsets_t get_reconstruct_offsets_neighbor_2d(const uint32_t i, 
                                                const uint32_t i_n,
                                                const uint8_t iface) const
  {

    offsets_t offsets;

    uint8_t ifaceX = iface>>IX;
    uint8_t ifaceY = iface>>IY;

    /*
     * - current cell and neighbor cell have the same size
     * or
     * - current cell is larger than neighbor
     */
    if (pmesh->getSize(i) >= pmesh->getSize(i_n)) {

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 1.0 - 2.0 * ifaceX;
        offsets[IY] = 0.0;
        offsets[IZ] = 0.0;
      }

      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = 0.0;
        offsets[IY] = 1.0 - 2.0 * ifaceY;
        offsets[IZ] = 0.0;
      }

    } // end same size

    /*
     * current cell is smaller than neighbor
     */
    if (pmesh->getSize(i) < pmesh->getSize(i_n)) {

      bitpit::darray3 xyz_c = pmesh->getCenter(i);
      bitpit::darray3 xyz_n = pmesh->getCenter(i_n);

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 1.0-2.0*ifaceX;
        offsets[IY] = xyz_n[IY]>xyz_c[IY] ? -0.5 : 0.5;
        offsets[IZ] = 0.0;
      }
      
      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = xyz_n[IX]>xyz_c[IX] ? 0.5 : -0.5;
        offsets[IY] = 1.0-2.0*ifaceY;
        offsets[IZ] = 0.0;
      }      

    } // end current cell is larger
  
    return offsets;

  } // get_reconstruct_offsets_neighbor_2d

  // =======================================================================
  // =======================================================================
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets.
   *
   * This is equivalent to trace operation in Ramses.
   * We just extrapolate primitive variables (at cell center) to border
   * using limited slopes.
   *
   * \note offsets are given in units dx/2.
   *
   * \param[in] q primitive variables at cell center
   * \param[in] offsets
   * \param[in] dx2 cell size divided by 2 (dx/2)
   *
   * \return qr reconstructed state (primitive variables)
   */
  KOKKOS_INLINE_FUNCTION
  HydroState2d reconstruct_state_2d(HydroState2d q, offsets_t offsets, real_t dx2) const
  {
    HydroState2d qr;
    
    return qr;
    
  } // reconstruct_state_2d

  // =======================================================================
  // =======================================================================
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

    // this vector contains quad ids
    // corresponding to neighbors
    std::vector<uint32_t> neigh; // through a given face

    // this vector contains ghost status of each neighbors
    std::vector<bool> isghost; // through a given face

    // current cell center state (primitive variables)
    HydroState2d qc;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qc[ivar] = Qdata(i,fm[ivar]);
    
    // iterate neighbors through a given face
    for (uint8_t iface = 0; iface < nfaces; ++iface) {
      
      // find neighbors Id
      pmesh->findNeighbours(i, iface, codim, neigh, isghost);
      
      // sweep neighbors accross face identified by iface
      for (uint16_t j = 0; j < neigh.size(); ++j) {

        uint32_t i_n = neigh[j];

        // 1. reconstruct on current cell side, two cases
        // - if neighbor is larger or same size, we reconstruct on face center
        // - if neighbor is smaller, we reconstruct at center of the sub-face

        // current cell reconstruction  (primitive variables)

        real_t dx_over_2 = pmesh->getSize(i);

        offsets_t offsets = get_reconstruct_offsets_current_2d(i, i_n, iface);
        HydroState2d qr_c = reconstruct_state_2d(qc, offsets, dx_over_2);

        // neighbor cell reconstruction (primitive variables)
        HydroState2d qr_n;

      } // end for j (neighbors accross a given face)

    } // end for iface

  } // operator_2d

  KOKKOS_INLINE_FUNCTION
  void operator_3d(const size_t i) const 
  {
  
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
  HydroParams  params;
  id2index_t   fm;
  DataArray    Data_in, Data_out;
  DataArray    Qdata;
  DataArray    SlopeX, SlopeY, SlopeZ;
  real_t       dt;
  
}; // ComputeFluxesAndUpdateHydroFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // COMPUTE_FLUXES_AND_UPDATE_HYDRO_FUNCTOR_H_
