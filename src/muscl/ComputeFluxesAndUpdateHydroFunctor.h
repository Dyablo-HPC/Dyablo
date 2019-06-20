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
#include "shared/RiemannSolvers.h"

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
   * Compute Riemann fluxes and update current cell only.
   *
   * \param[in]  pmesh pointer to AMR mesh structure
   * \param[in]  params
   * \param[in]  fm field map
   * \param[in]  Data_in current time step data (conservative variables)
   * \param[out] Data_out next time step data (conservative variables)
   * \param[in]  Qdata primitive variables
   * \param[in]  Qdata_ghost primitive variables in ghost cells
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
                                     DataArray Qdata_ghost,
                                     DataArray Slopes_x,
                                     DataArray Slopes_y,
                                     DataArray Slopes_z,
                                     real_t    dt) :
    HydroBaseFunctor(params),
    pmesh(pmesh),
    fm(fm),
    Data_in(Data_in),
    Data_out(Data_out),
    Qdata(Qdata),
    Qdata_ghost(Qdata_ghost),
    Slopes_x(Slopes_x),
    Slopes_y(Slopes_y),
    Slopes_z(Slopes_z),
    dt(dt)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams params,
		    id2index_t  fm,
		    DataArray Data_in,
		    DataArray Data_out,
                    DataArray Qdata,
		    DataArray Qdata_ghost,
		    DataArray SlopeX,
		    DataArray SlopeY,
		    DataArray SlopeZ,
                    real_t    dt)
  {
    ComputeFluxesAndUpdateHydroFunctor functor(pmesh, params, fm, 
                                               Data_in, Data_out,
                                               Qdata, Qdata_ghost,
                                               SlopeX,SlopeY,SlopeZ,
                                               dt);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  // =======================================================================
  // =======================================================================
  /**
   * a dummy swap device routine.
   */
  KOKKOS_INLINE_FUNCTION
  void swap ( real_t& a, real_t& b ) const {
    real_t c=a; a=b; b=c;
  } // swap
  
  // =======================================================================
  // =======================================================================
  template<uint8_t dir>
  KOKKOS_INLINE_FUNCTION
  bool face_along_axis(uint8_t iface) const
  {
    return ( iface>>1 == dir );

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
                                               const bool isghost_n,
                                               const uint8_t iface) const
  {

    offsets_t offsets;

    // iface2 :
    // - 0 for left  interface
    // - 1 for right interface
    uint8_t iface2 = iface & 0x1;

    /*
     * - current cell and neighbor cell have the same size
     * or
     * - current cell is smaller than neighbor
     */
    if (pmesh->getSize(i) <= pmesh->getSize(i_n)) {

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 2.0 * iface2 - 1;
        offsets[IY] = 0.0;
        offsets[IZ] = 0.0;
      }

      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = 0.0;
        offsets[IY] = 2.0 * iface2 - 1;
        offsets[IZ] = 0.0;
      }

    } // end same size

    /*
     * current cell is larger than neighbors
     */
    if (pmesh->getSize(i) > pmesh->getSize(i_n)) {

      bitpit::darray3 xyz_c = pmesh->getCenter(i);
      bitpit::darray3 xyz_n = pmesh->getCenter(i_n);
      
      // modify xyz_n if neighbor is actually a ghost cell
      if (isghost_n)
        xyz_n = pmesh->getCenterGhost(i_n);

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 2.0 * iface2 - 1;
        offsets[IY] = xyz_n[IY]>xyz_c[IY] ? 0.5 : -0.5;
        offsets[IZ] = 0.0;
      }
      
      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = xyz_n[IX]>xyz_c[IX] ? 0.5 : -0.5;
        offsets[IY] = 2.0 * iface2 - 1;
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
                                                const bool isghost_n,
                                                const uint8_t iface) const
  {

    offsets_t offsets;

    // iface2 :
    // - 0 for left  interface
    // - 1 for right interface
    uint8_t iface2 = iface & 0x1;

    /*
     * - current cell and neighbor cell have the same size
     * or
     * - current cell is larger than neighbor
     */
    if (pmesh->getSize(i) >= pmesh->getSize(i_n)) {

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 1.0 - 2.0 * iface2;
        offsets[IY] = 0.0;
        offsets[IZ] = 0.0;
      }

      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = 0.0;
        offsets[IY] = 1.0 - 2.0 * iface2;
        offsets[IZ] = 0.0;
      }

    } // end same size

    /*
     * current cell is smaller than neighbor
     */
    if (pmesh->getSize(i) < pmesh->getSize(i_n)) {

      bitpit::darray3 xyz_c = pmesh->getCenter(i);
      bitpit::darray3 xyz_n = pmesh->getCenter(i_n);

      // modify xyz_n if neighbor is actually a ghost cell
      if (isghost_n)
        xyz_n = pmesh->getCenterGhost(i_n);

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 1.0 -2.0 * iface2;
        offsets[IY] = xyz_n[IY]>xyz_c[IY] ? -0.5 : 0.5;
        offsets[IZ] = 0.0;
      }
      
      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = xyz_n[IX]>xyz_c[IX] ? -0.5 : 0.5;
        offsets[IY] = 1.0 - 2.0 * iface2;
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
   * \note How different is this with CanoP ? In CanoP reconstructed state are systematically taken
   *       at face center (from the point of view of the reconstructing cell) which may not correspond
   *       to the same location when current and neighbor cells have not the same size.
   *       Here we always reconstruct at the same location from side of an interface.
   *
   * \param[in] q primitive variables at cell center
   * \param[in] i cell id (needed to read slopes)
   * \param[in] offsets
   * \param[in] dx2 cell size divided by 2 (dx/2)
   * \param[in] dt
   *
   * \return qr reconstructed state (primitive variables)
   */
  KOKKOS_INLINE_FUNCTION
  HydroState2d reconstruct_state_2d(HydroState2d q, 
                                    uint32_t i,
                                    offsets_t offsets,
                                    real_t dx2, real_t dt) const
  {
    HydroState2d qr;
    
    const double gamma  = params.settings.gamma0;
    const double smallr = params.settings.smallr;
    
    //double xyz_center[3];
    
    real_t r,p,u,v,w;
    real_t sr0, sp0, su0, sv0, sw0;
    real_t drx, dpx, dux, dvx, dwx;
    real_t dry, dpy, duy, dvy, dwy;
    real_t drz, dpz, duz, dvz, dwz;
    
    const real_t dtdx = dt/(2*dx2);
    const real_t dtdy = dtdx;
    const real_t dtdz = params.dimType==THREE_D ? dtdx : 0.0;

    const real_t dy2 = dx2;

    // retrieve primitive variables in current quadrant
    r = q[ID];
    p = q[IP];
    u = q[IU];
    v = q[IV];
    w = 0.0;

    // retrieve variations = dx * slopes 
    drx = dx2 * Slopes_x(i,fm[ID]);
    dpx = dx2 * Slopes_x(i,fm[IP]);
    dux = dx2 * Slopes_x(i,fm[IU]);
    dvx = dx2 * Slopes_x(i,fm[IV]);
    dwx = 0.0;
    
    dry = dy2 * Slopes_y(i,fm[ID]);
    dpy = dy2 * Slopes_y(i,fm[IP]);
    duy = dy2 * Slopes_y(i,fm[IU]);
    dvy = dy2 * Slopes_y(i,fm[IV]);
    dwy = 0.0;
    
    drz = 0.0;
    dpz = 0.0;
    duz = 0.0;
    dvz = 0.0;
    dwz = 0.0;

    // source terms (with transverse derivatives)
    sr0 = (-u*drx-dux*r      )*dtdx + (-v*dry-dvy*r      )*dtdy + (-w*drz-dwz*r      )*dtdz;
    su0 = (-u*dux-dpx/r      )*dtdx + (-v*duy            )*dtdy + (-w*duz            )*dtdz;
    sv0 = (-u*dvx            )*dtdx + (-v*dvy-dpy/r      )*dtdy + (-w*dvz            )*dtdz;
    sw0 = (-u*dwx            )*dtdx + (-v*dwy            )*dtdy + (-w*dwz-dpz/r      )*dtdz;
    sp0 = (-u*dpx-dux*gamma*p)*dtdx + (-v*dpy-dvy*gamma*p)*dtdy + (-w*dpz-dwz*gamma*p)*dtdz;

    // Update in time the  primitive variables
    r = r + sr0;
    u = u + su0;
    v = v + sv0;
    w = w + sw0;
    p = p + sp0;
    
    // reconstruct state on interface
    qr[ID] = r + offsets[IX] * drx + offsets[IY] * dry;
    qr[IP] = p + offsets[IX] * dpx + offsets[IY] * dpy;
    qr[IU] = u + offsets[IX] * dux + offsets[IY] * duy ;
    qr[IV] = v + offsets[IX] * dvx + offsets[IY] * dvy ;
    qr[ID] = fmax(smallr, qr[ID]);

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

    // current cell size
    real_t dx = pmesh->getSize(i);
    
    // current cell volume
    real_t dV = dx*dx*dx;

    // this vector contains quad ids
    // corresponding to neighbors
    std::vector<uint32_t> neigh; // through a given face

    // this vector contains ghost status of each neighbors
    std::vector<bool> isghost; // through a given face

    // current cell center state (primitive variables)
    HydroState2d qprim;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qprim[ivar] = Qdata(i,fm[ivar]);

    // current cell conservative variable state
    HydroState2d qcons;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qcons[ivar] = Data_in(i,fm[ivar]);

    // iterate neighbors through a given face
    for (uint8_t iface = 0; iface < nfaces; ++iface) {
      
      // find neighbors Id
      pmesh->findNeighbours(i, iface, codim, neigh, isghost);
      
      // sweep neighbors accross face identified by iface
      for (uint16_t j = 0; j < neigh.size(); ++j) {

        uint32_t i_n = neigh[j];

        // 0. retrieve primitive variables in neighbor cell
        HydroState2d qprim_n;
        if (isghost[j]) {
          for (uint8_t ivar = 0; ivar < nbvar; ++ivar)
            qprim_n[ivar] = Qdata_ghost(i_n, fm[ivar]);
        } else {
          for (uint8_t ivar = 0; ivar < nbvar; ++ivar)
            qprim_n[ivar] = Qdata(i_n, fm[ivar]);
        }

        // 1. reconstruct primitive variables on both sides of current interface (iface)

        // current cell reconstruction  (primitive variables)
        const real_t dx_over_2 = pmesh->getSize(i);
        const offsets_t offsets = get_reconstruct_offsets_current_2d(i, i_n, isghost[j], iface);
        HydroState2d qr_c = reconstruct_state_2d(qprim, i, offsets, dx_over_2, dt);

        // neighbor cell reconstruction (primitive variables)
        const real_t dx_over_2_n = pmesh->getSize(i_n);
        const offsets_t offsets_n = get_reconstruct_offsets_neighbor_2d(i, i_n, isghost[j], iface);
        HydroState2d qr_n = reconstruct_state_2d(qprim_n, i_n, offsets_n, dx_over_2_n, dt);

        // 2. we now have "qleft / qright" state ready to solver Riemann problem
        HydroState2d flux;

        // riemann solver along Y direction requires to swap velocity
        // components
        if (face_along_axis<IY>(iface)) {
          swap(qr_c[IU], qr_c[IV]);
          swap(qr_n[IU], qr_n[IV]);
        }

        if (iface==0 or iface==2) {

          riemann_hydro(qr_n,qr_c,flux,params);

        } else if (iface==1 or iface==3) {

          riemann_hydro(qr_c,qr_n,flux,params);

        }

        // swap back velocity components in flux when dealing with 
        // a face along IY direction
        if (face_along_axis<IY>(iface)) {
          swap(flux[IU], flux[IV]);
        }
        
        // 3. accumulate flux into qcons
        
        // current face area:
        // if neighbor is smaller, flux is divided by the number of sub-faces
        // else only one interface (neigh.size = 1)
        real_t dS = dx*dx / neigh.size();
        real_t scale = dt*dS/dV;

        if (iface == 0 or iface == 2) {
          qcons[ID] += flux[ID]*scale;
          qcons[IE] += flux[IE]*scale;
          qcons[IU] += flux[IU]*scale;
          qcons[IV] += flux[IV]*scale;
        }
        if (iface == 1 or iface == 3) {
          qcons[ID] -= flux[ID]*scale;
          qcons[IE] -= flux[IE]*scale;
          qcons[IU] -= flux[IU]*scale;
          qcons[IV] -= flux[IV]*scale;
        }

      } // end for j (neighbors accross a given face)

    } // end for iface

    // Now we can update current cell (write qcons into Data_out)
    Data_out(i,fm[ID]) = qcons[ID];
    Data_out(i,fm[IE]) = qcons[IE];
    Data_out(i,fm[IU]) = qcons[IU];
    Data_out(i,fm[IV]) = qcons[IV];

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
  id2index_t   fm;
  DataArray    Data_in, Data_out;
  DataArray    Qdata, Qdata_ghost;
  DataArray    Slopes_x, Slopes_y, Slopes_z;
  real_t       dt;
  
}; // ComputeFluxesAndUpdateHydroFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // COMPUTE_FLUXES_AND_UPDATE_HYDRO_FUNCTOR_H_
