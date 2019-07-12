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
#include "shared/bc_utils.h"

// base class
#include "muscl/HydroBaseFunctor.h"

namespace dyablo { namespace muscl {

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
   * \param[in]  Slopes_x_ghost limited slopes along x axis, ghost cells
   * \param[in]  Slopes_y_ghost limited slopes along y axis, ghost cells
   * \param[in]  Slopes_z_ghost limited slopes along z axis, ghost cells
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
                                     DataArray Slopes_x_ghost,
                                     DataArray Slopes_y_ghost,
                                     DataArray Slopes_z_ghost,
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
    Slopes_x_ghost(Slopes_x_ghost),
    Slopes_y_ghost(Slopes_y_ghost),
    Slopes_z_ghost(Slopes_z_ghost),
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
		    DataArray SlopeX_ghost,
		    DataArray SlopeY_ghost,
		    DataArray SlopeZ_ghost,
                    real_t    dt)
  {
    ComputeFluxesAndUpdateHydroFunctor functor(pmesh, params, fm, 
                                               Data_in, Data_out,
                                               Qdata, Qdata_ghost,
                                               SlopeX,SlopeY,SlopeZ,
                                               SlopeX_ghost,SlopeY_ghost,SlopeZ_ghost,
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
   * \param[in] isghost_n boolean status of neighbor cell (ghost or regular ?)
   * \param[in] iface face id (from current cell side point of view)
   */
  KOKKOS_INLINE_FUNCTION
  offsets_t get_reconstruct_offsets_current(const uint32_t i, 
                                            const uint32_t i_n,
                                            const bool isghost_n,
                                            const uint8_t iface) const
  {

    // get dimension
    const int dim = this->params.dimType == TWO_D ? 2 : 3;

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
    const real_t size_c = pmesh->getSize(i);
    const real_t size_n = isghost_n ? 
      pmesh->getSizeGhost(i_n) : 
      pmesh->getSize(i_n);


    /*
     * current cell (c) is smaller or has same size as neighbor (n)
     *
     * +----+          +----+----+
     * | n  |__    or  | n  | c  |
     * |    | c|       |    |    |
     * +----+--+       +----+----+
     *
     */
    if (size_c <= size_n) {

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

      // along y axis
      if (face_along_axis<IZ>(iface)) {
        offsets[IX] = 0.0;
        offsets[IY] = 0.0;
        offsets[IZ] = 2.0 * iface2 - 1;
      }

    } // end same size

    /*
     * current cell (c) is larger than neighbor (n)
     *
     *    +----+
     *  __| c  |
     * | n|    |
     * +--+----+
     *
     */
    if (size_c > size_n) {

      const bitpit::darray3 xyz_c = pmesh->getCenter(i);
      const bitpit::darray3 xyz_n = isghost_n ? 
        pmesh->getCenterGhost(i_n) :
        pmesh->getCenter(i_n);

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 2.0 * iface2 - 1;
        offsets[IY] = xyz_n[IY]>xyz_c[IY] ? 0.5 : -0.5;
        offsets[IZ] = dim==2 ? 0.0 : (xyz_n[IZ]>xyz_c[IZ] ? 0.5 : -0.5) ;
      }
      
      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = xyz_n[IX]>xyz_c[IX] ? 0.5 : -0.5;
        offsets[IY] = 2.0 * iface2 - 1;
        offsets[IZ] = dim==2 ? 0.0 : (xyz_n[IZ]>xyz_c[IZ] ? 0.5 : -0.5) ;
      }      

      // along z axis
      if (face_along_axis<IZ>(iface)) {
        offsets[IX] = xyz_n[IX]>xyz_c[IX] ? 0.5 : -0.5;
        offsets[IY] = xyz_n[IY]>xyz_c[IY] ? 0.5 : -0.5;
        offsets[IZ] = 2.0 * iface2 - 1;
      }

    } // end current cell is larger

    return offsets;

  } // get_reconstruct_offsets_current

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
  offsets_t get_reconstruct_offsets_neighbor(const uint32_t i, 
                                             const uint32_t i_n,
                                             const bool isghost_n,
                                             const uint8_t iface) const
  {

    // get dimension
    const int dim = this->params.dimType == TWO_D ? 2 : 3;

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
    const real_t size_c = pmesh->getSize(i);
    const real_t size_n = isghost_n ? 
      pmesh->getSizeGhost(i_n) : 
      pmesh->getSize(i_n);

    /*
     * current cell (c) is larger than neighbor (n) or same size
     *
     *    +----+       +----+----+
     *  __| c  |   or  | n  | c  |
     * | n|    |       |    |    |
     * +--+----+       +----+----+
     *
     */
    if (size_c >= size_n) {

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

      // along z axis
      if (face_along_axis<IZ>(iface)) {
        offsets[IX] = 0.0;
        offsets[IY] = 0.0;
        offsets[IZ] = 1.0 - 2.0 * iface2;
      }

    } // end same size

    /*
     * current cell (c) is smaller than neighbor (n)
     *
     * +----+    
     * | n  |__  
     * |    | c| 
     * +----+--+ 
     *
     */
    if (size_c < size_n) {

      const bitpit::darray3 xyz_c = pmesh->getCenter(i);
      const bitpit::darray3 xyz_n = isghost_n ? 
        pmesh->getCenterGhost(i_n) : 
        pmesh->getCenter(i_n);      

      // along x axis
      if (face_along_axis<IX>(iface)) {
        offsets[IX] = 1.0 - 2.0 * iface2;
        offsets[IY] = xyz_n[IY]>xyz_c[IY] ? -0.5 : 0.5;
        offsets[IZ] = dim==2 ? 0.0 : (xyz_n[IZ]>xyz_c[IZ] ? -0.5 : 0.5) ;
      }
      
      // along y axis
      if (face_along_axis<IY>(iface)) {
        offsets[IX] = xyz_n[IX]>xyz_c[IX] ? -0.5 : 0.5;
        offsets[IY] = 1.0 - 2.0 * iface2;
        offsets[IZ] = dim==2 ? 0.0 : (xyz_n[IZ]>xyz_c[IZ] ? -0.5 : 0.5) ;
      }

      // along y axis
      if (face_along_axis<IZ>(iface)) {
        offsets[IX] = xyz_n[IX]>xyz_c[IX] ? -0.5 : 0.5;
        offsets[IY] = xyz_n[IY]>xyz_c[IY] ? -0.5 : 0.5;
        offsets[IZ] = 1.0 - 2.0 * iface2;
      }

    } // end current cell is larger
 
    return offsets;

  } // get_reconstruct_offsets_neighbor

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
   * \param[in] isghost boolean stating if the cell where reconstruction happens is a ghost cell
   * \param[in] offsets
   * \param[in] dx_over_2 cell size divided by 2 (dx/2)
   * \param[in] dt
   *
   * \return qr reconstructed state (primitive variables)
   */
  KOKKOS_INLINE_FUNCTION
  HydroState2d reconstruct_state_2d(HydroState2d q, 
                                    uint32_t i,
                                    bool isghost,
                                    offsets_t offsets,
                                    real_t dx_over_2, real_t dt) const
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
    
    const real_t dtdx = dt/(2*dx_over_2);
    const real_t dtdy = dtdx;
    const real_t dtdz = params.dimType==THREE_D ? dtdx : 0.0;

    real_t &dx2 = dx_over_2;
    real_t &dy2 = dx_over_2;

    // retrieve primitive variables in current quadrant
    r = q[ID];
    p = q[IP];
    u = q[IU];
    v = q[IV];
    w = 0.0;

    // retrieve variations = dx * slopes
    if (isghost) {
      drx = dx2 * Slopes_x_ghost(i, fm[ID]);
      dpx = dx2 * Slopes_x_ghost(i, fm[IP]);
      dux = dx2 * Slopes_x_ghost(i, fm[IU]);
      dvx = dx2 * Slopes_x_ghost(i, fm[IV]);
      dwx = 0.0;

      dry = dy2 * Slopes_y_ghost(i, fm[ID]);
      dpy = dy2 * Slopes_y_ghost(i, fm[IP]);
      duy = dy2 * Slopes_y_ghost(i, fm[IU]);
      dvy = dy2 * Slopes_y_ghost(i, fm[IV]);
      dwy = 0.0;

      drz = 0.0;
      dpz = 0.0;
      duz = 0.0;
      dvz = 0.0;
      dwz = 0.0;
    } else {
      drx = dx2 * Slopes_x(i, fm[ID]);
      dpx = dx2 * Slopes_x(i, fm[IP]);
      dux = dx2 * Slopes_x(i, fm[IU]);
      dvx = dx2 * Slopes_x(i, fm[IV]);
      dwx = 0.0;

      dry = dy2 * Slopes_y(i, fm[ID]);
      dpy = dy2 * Slopes_y(i, fm[IP]);
      duy = dy2 * Slopes_y(i, fm[IU]);
      dvy = dy2 * Slopes_y(i, fm[IV]);
      dwy = 0.0;

      drz = 0.0;
      dpz = 0.0;
      duz = 0.0;
      dvz = 0.0;
      dwz = 0.0;
    }

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
  /**
   * Reconstruct an hydro state at a cell border location specified by offsets (3d version).
   *
   * \sa reconstruct_state_2d
   */
  KOKKOS_INLINE_FUNCTION
  HydroState3d reconstruct_state_3d(HydroState3d q, 
                                    uint32_t i,
                                    bool isghost,
                                    offsets_t offsets,
                                    real_t dx_over_2, real_t dt) const
  {
    HydroState3d qr;
    
    const double gamma  = params.settings.gamma0;
    const double smallr = params.settings.smallr;
    
    //double xyz_center[3];
    
    real_t r,p,u,v,w;
    real_t sr0, sp0, su0, sv0, sw0;
    real_t drx, dpx, dux, dvx, dwx;
    real_t dry, dpy, duy, dvy, dwy;
    real_t drz, dpz, duz, dvz, dwz;
    
    const real_t dtdx = dt/(2*dx_over_2);
    const real_t dtdy = dtdx;
    const real_t dtdz = dtdx;

    real_t &dx2 = dx_over_2;
    real_t &dy2 = dx_over_2;
    real_t &dz2 = dx_over_2;

    // retrieve primitive variables in current quadrant
    r = q[ID];
    p = q[IP];
    u = q[IU];
    v = q[IV];
    w = q[IW];

    // retrieve variations = dx * slopes
    if (isghost) {

      drx = dx2 * Slopes_x_ghost(i, fm[ID]);
      dpx = dx2 * Slopes_x_ghost(i, fm[IP]);
      dux = dx2 * Slopes_x_ghost(i, fm[IU]);
      dvx = dx2 * Slopes_x_ghost(i, fm[IV]);
      dwx = dx2 * Slopes_x_ghost(i, fm[IW]);

      dry = dy2 * Slopes_y_ghost(i, fm[ID]);
      dpy = dy2 * Slopes_y_ghost(i, fm[IP]);
      duy = dy2 * Slopes_y_ghost(i, fm[IU]);
      dvy = dy2 * Slopes_y_ghost(i, fm[IV]);
      dwy = dy2 * Slopes_y_ghost(i, fm[IW]);

      drz = dz2 * Slopes_z_ghost(i, fm[ID]);
      dpz = dz2 * Slopes_z_ghost(i, fm[IP]);
      duz = dz2 * Slopes_z_ghost(i, fm[IU]);
      dvz = dz2 * Slopes_z_ghost(i, fm[IV]);
      dwz = dz2 * Slopes_z_ghost(i, fm[IW]);

    } else {

      drx = dx2 * Slopes_x(i, fm[ID]);
      dpx = dx2 * Slopes_x(i, fm[IP]);
      dux = dx2 * Slopes_x(i, fm[IU]);
      dvx = dx2 * Slopes_x(i, fm[IV]);
      dwx = dx2 * Slopes_x(i, fm[IW]);

      dry = dy2 * Slopes_y(i, fm[ID]);
      dpy = dy2 * Slopes_y(i, fm[IP]);
      duy = dy2 * Slopes_y(i, fm[IU]);
      dvy = dy2 * Slopes_y(i, fm[IV]);
      dwy = dy2 * Slopes_y(i, fm[IW]);

      drz = dz2 * Slopes_z(i, fm[ID]);
      dpz = dz2 * Slopes_z(i, fm[IP]);
      duz = dz2 * Slopes_z(i, fm[IU]);
      dvz = dz2 * Slopes_z(i, fm[IV]);
      dwz = dz2 * Slopes_z(i, fm[IW]);

    }

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
    qr[ID] = r + offsets[IX] * drx + offsets[IY] * dry + offsets[IZ] * drz ;
    qr[IP] = p + offsets[IX] * dpx + offsets[IY] * dpy + offsets[IZ] * dpz ;
    qr[IU] = u + offsets[IX] * dux + offsets[IY] * duy + offsets[IZ] * duz ;
    qr[IV] = v + offsets[IX] * dvx + offsets[IY] * dvy + offsets[IZ] * dvz ;
    qr[IW] = w + offsets[IX] * dwx + offsets[IY] * dwy + offsets[IZ] * dwz ;

    qr[ID] = fmax(smallr, qr[ID]);

    return qr;
    
  } // reconstruct_state_3d

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

      //===================================================
      //
      // Border conditions: define reconstructed states on both
      // sides of an interface at external border
      //
      // is current cell touching the external border ?
      //===================================================
      if (neigh.size()==0) {

        HydroState2d qr_c, qr_n; 
        
        // take care of border conditions (in case of open or
        //reflective border)
        
        // get x,y,z coordinate at current cell center
        const bitpit::darray3 xyz_c = pmesh->getCenter(i);
        const double &x = xyz_c[IX];
        const double &y = xyz_c[IY];
        
        if ( is_at_border<XMIN>(dx,x) and iface == 0 ) {
          if (params.boundary_type_xmin == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_xmin == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IU] = -qr_n[IU];
          }
        }
        
        if ( is_at_border<XMAX>(dx,x) and iface == 1 ) {
          if (params.boundary_type_xmax == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_xmax == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IU] = -qr_n[IU];
          }
        }
          
        if ( is_at_border<YMIN>(dx,y) and iface == 2 ) {
          if (params.boundary_type_ymin == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_ymin == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IV] = -qr_n[IV];
          }
        }
        
        if ( is_at_border<YMAX>(dx,y) and iface == 3 ) {
          if (params.boundary_type_ymax == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_ymax == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IV] = -qr_n[IV];
          }
        }

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
        real_t dS = dx*dx;
        real_t scale = dt*dS/dV;

        // iface = 0 or 2
        if ( (iface & 0x1) == 0 ) {
          qcons[ID] += flux[ID]*scale;
          qcons[IE] += flux[IE]*scale;
          qcons[IU] += flux[IU]*scale;
          qcons[IV] += flux[IV]*scale;
        } else {
          qcons[ID] -= flux[ID]*scale;
          qcons[IE] -= flux[IE]*scale;
          qcons[IU] -= flux[IU]*scale;
          qcons[IV] -= flux[IV]*scale;
        }
      
      } // end neigh.size == 0

      //===================================================
      // Deal with bulk cells (no face with zero neighbors)
      //
      // sweep neighbors accross face identified by iface
      //===================================================
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
        const real_t dx_over_2 = pmesh->getSize(i)/2;
        const offsets_t offsets = get_reconstruct_offsets_current(i, i_n, isghost[j], iface);
        HydroState2d qr_c = reconstruct_state_2d(qprim, i, false, offsets, dx_over_2, dt);

        // neighbor cell reconstruction (primitive variables)
        const real_t size_n = isghost[j] ? pmesh->getSizeGhost(i_n) : pmesh->getSize(i_n);
        const real_t dx_over_2_n = size_n/2;
        const offsets_t offsets_n = get_reconstruct_offsets_neighbor(i, i_n, isghost[j], iface);
        HydroState2d qr_n = reconstruct_state_2d(qprim_n, i_n, isghost[j], offsets_n, dx_over_2_n, dt);
        

        // 2. we now have "qleft / qright" state ready to solver Riemann problem
        HydroState2d flux;

        // riemann solver along Y direction requires to swap velocity
        // components
        if (face_along_axis<IY>(iface)) {
          swap(qr_c[IU], qr_c[IV]);
          swap(qr_n[IU], qr_n[IV]);
        }

        // iface = 0 or 2
        if ( (iface & 0x1) == 0 ) {

          riemann_hydro(qr_n,qr_c,flux,params);

        } else {

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

        // iface = 0 or 2
        if ( (iface & 0x1) == 0 ) {
          qcons[ID] += flux[ID]*scale;
          qcons[IE] += flux[IE]*scale;
          qcons[IU] += flux[IU]*scale;
          qcons[IV] += flux[IV]*scale;
        } else { // iface = 1 or 3
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

  // =======================================================================
  // =======================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_3d(const size_t i) const 
  {
  
    constexpr int dim = 3;
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
    HydroState3d qprim;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qprim[ivar] = Qdata(i,fm[ivar]);

    // current cell conservative variable state
    HydroState3d qcons;
    for (uint8_t ivar=0; ivar<nbvar; ++ivar)
      qcons[ivar] = Data_in(i,fm[ivar]);

    // iterate neighbors through a given face
    for (uint8_t iface = 0; iface < nfaces; ++iface) {
      
      // find neighbors Id
      pmesh->findNeighbours(i, iface, codim, neigh, isghost);

      //===================================================
      //
      // Border conditions: define reconstructed states on both
      // sides of an interface at external border
      //
      // is current cell touching the external border ?
      //===================================================
      if (neigh.size() == 0) {

        HydroState3d qr_c, qr_n;

        // take care of border conditions (in case of open or
        // reflective border)
        
        // get x,y,z coordinate at current cell center
        const bitpit::darray3 xyz_c = pmesh->getCenter(i);
        const double &x = xyz_c[IX];
        const double &y = xyz_c[IY];
        const double &z = xyz_c[IZ];

        if ( is_at_border<XMIN>(dx,x) and iface == 0 ) {
          if (params.boundary_type_xmin == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_xmin == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IU] = -qr_n[IU];
          }
        }
        
        if ( is_at_border<XMAX>(dx,x) and iface == 1 ) {
          if (params.boundary_type_xmax == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_xmax == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IU] = -qr_n[IU];
          }
        }
          
        if ( is_at_border<YMIN>(dx,y) and iface == 2 ) {
          if (params.boundary_type_ymin == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_ymin == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IV] = -qr_n[IV];
          }
        }
        
        if ( is_at_border<YMAX>(dx,y) and iface == 3 ) {
          if (params.boundary_type_ymax == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_ymax == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IV] = -qr_n[IV];
          }
        }

        if ( is_at_border<ZMIN>(dx,z) and iface == 4 ) {
          if (params.boundary_type_zmin == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_zmin == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IW] = -qr_n[IW];
          }
        }
        
        if ( is_at_border<ZMAX>(dx,z) and iface == 5 ) {
          if (params.boundary_type_zmax == BC_ABSORBING) {
            qr_n = qprim;
            qr_c = qprim;
          }
          if (params.boundary_type_zmax == BC_REFLECTING) {
            qr_n = qprim;
            qr_c = qprim;
            qr_n[IW] = -qr_n[IW];
          }
        }

        // 2. we now have "qleft / qright" state ready to solver Riemann problem
        HydroState3d flux;

        // riemann solver along Y or Z direction requires to 
        // swap velocity components
        if (face_along_axis<IY>(iface)) {
          swap(qr_c[IU], qr_c[IV]);
          swap(qr_n[IU], qr_n[IV]);
        }
        if (face_along_axis<IZ>(iface)) {
          swap(qr_c[IU], qr_c[IW]);
          swap(qr_n[IU], qr_n[IW]);
        }

        // iface = 0, 2 or 4
        if ( (iface & 0x1) == 0 ) {

          riemann_hydro(qr_n,qr_c,flux,params);

        } else {

          riemann_hydro(qr_c,qr_n,flux,params);

        }

        // swap back velocity components in flux when dealing with 
        // a face along IY or IZ direction
        if (face_along_axis<IY>(iface)) {
          swap(flux[IU], flux[IV]);
        }
        if (face_along_axis<IZ>(iface)) {
          swap(flux[IU], flux[IW]);
        }
        
        // 3. accumulate flux into qcons
        
        // current face area:
        // if neighbor is smaller, flux is divided by the number of sub-faces
        // else only one interface (neigh.size = 1)
        real_t dS = dx*dx;
        real_t scale = dt*dS/dV;

        // iface = 0, 2 or 4
        if ( (iface & 0x1) == 0 ) {
          qcons[ID] += flux[ID]*scale;
          qcons[IE] += flux[IE]*scale;
          qcons[IU] += flux[IU]*scale;
          qcons[IV] += flux[IV]*scale;
          qcons[IW] += flux[IW]*scale;
        } else {
          qcons[ID] -= flux[ID]*scale;
          qcons[IE] -= flux[IE]*scale;
          qcons[IU] -= flux[IU]*scale;
          qcons[IV] -= flux[IV]*scale;
          qcons[IW] -= flux[IW]*scale;
        }

      } // end neigh.size() == 0

      //===================================================
      // Deal with bulk cells (no face with zero neighbors)
      //
      // sweep neighbors accross face identified by iface
      //===================================================
      for (uint16_t j = 0; j < neigh.size(); ++j) {

        uint32_t i_n = neigh[j];

        // 0. retrieve primitive variables in neighbor cell
        HydroState3d qprim_n;
        if (isghost[j]) {
          for (uint8_t ivar = 0; ivar < nbvar; ++ivar)
            qprim_n[ivar] = Qdata_ghost(i_n, fm[ivar]);
        } else {
          for (uint8_t ivar = 0; ivar < nbvar; ++ivar)
            qprim_n[ivar] = Qdata(i_n, fm[ivar]);
        }

        // 1. reconstruct primitive variables on both sides of current interface (iface)

        // current cell reconstruction  (primitive variables)
        const real_t dx_over_2 = pmesh->getSize(i)/2;
        const offsets_t offsets = get_reconstruct_offsets_current(i, i_n, isghost[j], iface);
        HydroState3d qr_c = reconstruct_state_3d(qprim, i, false, offsets, dx_over_2, dt);

        // neighbor cell reconstruction (primitive variables)
        const real_t size_n = isghost[j] ? pmesh->getSizeGhost(i_n) : pmesh->getSize(i_n);
        const real_t dx_over_2_n = size_n/2;
        const offsets_t offsets_n = get_reconstruct_offsets_neighbor(i, i_n, isghost[j], iface);
        HydroState3d qr_n = reconstruct_state_3d(qprim_n, i_n, isghost[j], offsets_n, dx_over_2_n, dt);

        // 2. we now have "qleft / qright" state ready to solver Riemann problem
        HydroState3d flux;

        // riemann solver along Y or Z direction requires to swap velocity
        // components
        if (face_along_axis<IY>(iface)) {
          swap(qr_c[IU], qr_c[IV]);
          swap(qr_n[IU], qr_n[IV]);
        }
        if (face_along_axis<IZ>(iface)) {
          swap(qr_c[IU], qr_c[IW]);
          swap(qr_n[IU], qr_n[IW]);
        }

        // iface = 0, 2 or 4
        if ( (iface & 0x1) == 0 ) {

          riemann_hydro(qr_n,qr_c,flux,params);

        } else { // iface = 1, 3 or 5

          riemann_hydro(qr_c,qr_n,flux,params);

        }

        // swap back velocity components in flux when dealing with
        // a face along IY or IZ direction
        if (face_along_axis<IY>(iface)) {
          swap(flux[IU], flux[IV]);
        }
        if (face_along_axis<IZ>(iface)) {
          swap(flux[IU], flux[IW]);
        }
        
        // 3. accumulate flux into qcons
        
        // current face area:
        // if neighbor is smaller, flux is divided by the number of sub-faces
        // else only one interface (neigh.size = 1)
        real_t dS = dx*dx / neigh.size();
        real_t scale = dt*dS/dV;

        // iface = 0, 2 or 4
        if ( (iface & 0x1) == 0 ) {
          qcons[ID] += flux[ID]*scale;
          qcons[IE] += flux[IE]*scale;
          qcons[IU] += flux[IU]*scale;
          qcons[IV] += flux[IV]*scale;
          qcons[IW] += flux[IW]*scale;
        } else {
          qcons[ID] -= flux[ID]*scale;
          qcons[IE] -= flux[IE]*scale;
          qcons[IU] -= flux[IU]*scale;
          qcons[IV] -= flux[IV]*scale;
          qcons[IW] -= flux[IW]*scale;
        }

      } // end for j (neighbors accross a given face)

    } // end for iface

    // Now we can update current cell (write qcons into Data_out)
    Data_out(i,fm[ID]) = qcons[ID];
    Data_out(i,fm[IE]) = qcons[IE];
    Data_out(i,fm[IU]) = qcons[IU];
    Data_out(i,fm[IV]) = qcons[IV];
    Data_out(i,fm[IW]) = qcons[IW];

  } // operator_3d

  // =======================================================================
  // =======================================================================
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
  DataArray    Slopes_x_ghost, Slopes_y_ghost, Slopes_z_ghost;
  real_t       dt;
  
}; // ComputeFluxesAndUpdateHydroFunctor

} // namespace muscl

} // namespace dyablo

#endif // COMPUTE_FLUXES_AND_UPDATE_HYDRO_FUNCTOR_H_
