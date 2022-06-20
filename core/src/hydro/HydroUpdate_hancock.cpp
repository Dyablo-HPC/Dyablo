#include "HydroUpdate_base.h"
#include "states/State_forward.h"
#include "utils/monitoring/Timers.h"

#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "RiemannSolvers.h"

#include "HydroUpdate_utils.h"


namespace dyablo { 

namespace {
  using GhostedArray = ForeachCell::CellArray_global_ghosted;
  using GlobalArray = ForeachCell::CellArray_global;
  using PatchArray = ForeachCell::CellArray_patch;
  using CellIndex = ForeachCell::CellIndex;

  /**
   * @brief Computes the source term from the muscl-hancock algorithm
   * 
   * @tparam ndim The number of dimensions
   * @tparam PrimState The primitive state variable being manipulated
   * @param q Centered primitive variable
   * @param slopeX Slopes along each direction
   * @param slopeY 
   * @param slopeZ 
   * @param dtdx time step over space step along each direction
   * @param dtdy 
   * @param dtdz 
   * @param gamma adiabatic index
   * @return A primitive state corresponding to the half-step evolved variable in the cell
   * 
   * @todo Adapt this to any PrimState possible !
   */
  template<int ndim, typename PrimState >
  KOKKOS_INLINE_FUNCTION
  PrimState compute_source( const PrimState& q,
                            const PrimState& slopeX,
                            const PrimState& slopeY,
                            const PrimState& slopeZ,
                            real_t dtdx, real_t dtdy, real_t dtdz,
                            real_t gamma )
  {
    // retrieve primitive variables in current quadrant
    const real_t r = q.rho;
    const real_t p = q.p;
    const real_t u = q.u;
    const real_t v = q.v;
    const real_t w = q.w;

    // retrieve variations = dx * slopes
    const real_t drx = slopeX.rho * 0.5;
    const real_t dpx = slopeX.p   * 0.5;
    const real_t dux = slopeX.u   * 0.5;
    const real_t dvx = slopeX.v   * 0.5;
    const real_t dwx = slopeX.w   * 0.5;    
    const real_t dry = slopeY.rho * 0.5;
    const real_t dpy = slopeY.p   * 0.5;
    const real_t duy = slopeY.u   * 0.5;
    const real_t dvy = slopeY.v   * 0.5;
    const real_t dwy = slopeY.w   * 0.5;    
    const real_t drz = slopeZ.rho * 0.5;
    const real_t dpz = slopeZ.p   * 0.5;
    const real_t duz = slopeZ.u   * 0.5;
    const real_t dvz = slopeZ.v   * 0.5;
    const real_t dwz = slopeZ.w   * 0.5;

    PrimState source{};
    if( ndim == 3 )
    {
      source.rho = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy + (-w * drz - dwz * r) * dtdz;
      source.u   = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy + (-w * duz) * dtdz;
      source.v   = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy + (-w * dvz) * dtdz;
      source.w   = w + (-u * dwx) * dtdx + (-v * dwy) * dtdy + (-w * dwz - dpz / r) * dtdz;
      source.p   = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy + (-w * dpz - dwz * gamma * p) * dtdz;
    }
    else
    {
      source.rho = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
      source.u   = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
      source.v   = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
      source.p   = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;
    }
    return source;
  }

  /**
   * @brief Computes slopes and sources along each directions on a cell 
   * 
   * @tparam ndim the number of dimensions 
   * @tparam State the current state being manipulated
   * @param Qgroup Array of primitive variables
   * @param iCell_Sources Index of the current cell in Qgroup
   * @param slope_type type of slope to compute
   * @param gamma adiabatic index
   * @param dx space-step
   * @param dy 
   * @param dz 
   * @param dt time-step
   * @param Sources (OUT) Source array
   * @param SlopesX (OUT) Slopes arrays
   * @param SlopesY 
   * @param SlopesZ 
   */
  template < 
    int ndim,
    typename State >
  KOKKOS_INLINE_FUNCTION
  void compute_slopes( const PatchArray& Qgroup, const CellIndex& iCell_Sources, int slope_type, real_t gamma, real_t dx, real_t dy, real_t dz, real_t dt,
                       const PatchArray& Sources, const PatchArray& SlopesX, const PatchArray& SlopesY, const PatchArray& SlopesZ)
  {
    using o_t = typename CellIndex::offset_t;
    using PrimState = typename State::PrimState;

    PrimState sx{}, sy{}, sz{};

    CellIndex ib = Qgroup.convert_index(iCell_Sources);
    PrimState qc;
    getPrimitiveState<ndim>( Qgroup, ib, qc );

    {
      // neighbor along x axis
      PrimState qm, qp;
      getPrimitiveState<ndim>( Qgroup, ib + o_t{-1, 0, 0}, qm);
      getPrimitiveState<ndim>( Qgroup, ib + o_t{ 1, 0, 0}, qp);     

      sx = compute_slope<ndim>(qm, qc, qp, 1.0, 1.0);
      CellIndex iCell_x = SlopesX.convert_index_ghost(iCell_Sources);
      if(iCell_x.is_valid())
        setPrimitiveState<ndim>(SlopesX, iCell_x, sx);
    }

    {
      // neighbor along y axis
      PrimState qm, qp;
      getPrimitiveState<ndim>( Qgroup, ib + o_t{ 0,-1, 0}, qm);
      getPrimitiveState<ndim>( Qgroup, ib + o_t{ 0, 1, 0}, qp);       

      sy = compute_slope<ndim>(qm, qc, qp, 1.0, 1.0);

      CellIndex iCell_y = SlopesY.convert_index_ghost(iCell_Sources);
      if(iCell_y.is_valid())
        setPrimitiveState<ndim>(SlopesY, iCell_y, sy);
    }

    if( ndim == 3 )
    {      
      // neighbor along z axis
      PrimState qm, qp;
      getPrimitiveState<ndim>( Qgroup, ib + o_t{ 0, 0,-1}, qm);
      getPrimitiveState<ndim>( Qgroup, ib + o_t{ 0, 0, 1}, qp);       

      sz = compute_slope<ndim>(qm, qc, qp, 1.0, 1.0);

      CellIndex iCell_z = SlopesZ.convert_index_ghost(iCell_Sources);
      if(iCell_z.is_valid())
        setPrimitiveState<ndim>(SlopesZ, iCell_z, sz);
    }

    {
      PrimState source = compute_source<ndim>(qc, sx, sy, sz, dt/dx, dt/dy, dt/dz, gamma);
      setPrimitiveState<ndim>(Sources, iCell_Sources, source);
    }
  }

  /**
   * @brief Reconstructs a MUSCL-HANCOCK state according to a source term and a slope
   * 
   * @tparam PrimState The type of primitive variable being manipulated
   * @param source The source term
   * @param slope the slope along the current direction
   * @param sign -1 if we reconstruct on the left, +1 if we reconstruct on the right
   * @param smallr minimum tolerated value for the density
   * @return KOKKOS_INLINE_FUNCTION 
   */
  template < typename PrimState >
  KOKKOS_INLINE_FUNCTION
  PrimState reconstruct_state(const PrimState& source, 
                              const PrimState& slope,
                              real_t sign, real_t smallr )
  {
    PrimState res = source + sign * slope * 0.5;
    res.rho = fmax(smallr, res.rho);

    return res;
  }

  /**
   * @brief Computes a flux between two states.
   * 
   * @tparam ndim the number of dimensions
   * @tparam State the current state being manipulated
   * @param sourceL left source term
   * @param sourceR right source term
   * @param slopeL left slope term
   * @param slopeR right slope term
   * @param dir the direction normal to the interface
   * @param smallr minimum tolerated value for the density
   * @param params parameters to feed to the Riemann solver
   * @return the conservative variable flux
   */
  template< 
    int ndim, 
    typename State >
  KOKKOS_INLINE_FUNCTION
  typename State::ConsState compute_flux( const typename State::PrimState& sourceL, const typename State::PrimState& sourceR, 
                                          const typename State::PrimState& slopeL,  const typename State::PrimState& slopeR,
                                          ComponentIndex3D dir, real_t smallr, const RiemannParams& params )
  {
    using PrimState = typename State::PrimState;
    using ConsState = typename State::ConsState;

    PrimState qL = reconstruct_state( sourceL, slopeL, 1, smallr );
    PrimState qR = reconstruct_state( sourceR, slopeR, -1, smallr );

    // Riemann solver along Y or Z direction requires to 
    // swap velocity components
    if (dir == IY) {
      swap(qL.u, qL.v);
      swap(qR.u, qR.v);
    }
    else if (dir == IZ) {
      swap(qL.u, qL.w);
      swap(qR.u, qR.w);
    }

    // Compute flux (Riemann solver)
    ConsState flux = riemann_hydro(qL, qR, params);
    
    if (dir == IY)
      swap(flux.rho_u, flux.rho_v);
    else if (dir == IZ)
      swap(flux.rho_u, flux.rho_w);

    return flux;
  }

  /**
   * @brief Compute both fluxes (using Riemann solver) in a direction for one cell and update U2
   * 
   * @tparam ndim the number of dimensions 
   * @tparam State the current state being manipulated
   * @param dir the direction orthogonal to the interface
   * @param iCell_Uout the index of the cell being modified
   * @param Slopes an array of slopes along the direction
   * @param Source an array of source terms
   * @param params the parameters to feed to the Riemann solver
   * @param smallr minimum tolerated value for the density
   * @param dtddir time-step over space-step along dir
   * @param Uout the array to update
   * 
   * @note this version is non-conservative because non-conforming interfaces are not taken into account
   */
  template < 
    int ndim,
    typename State>
  KOKKOS_INLINE_FUNCTION
  void compute_fluxes(ComponentIndex3D dir, 
                      const CellIndex& iCell_Uout, 
                      const PatchArray& Slopes,
                      const PatchArray& Source,
                      const RiemannParams& params,
                      real_t smallr,
                      real_t dtddir,
                      const GlobalArray& Uout
                      )
  {
    using PrimState = typename State::PrimState;
    using ConsState = typename State::ConsState;

    typename CellIndex::offset_t offsetm = {};
    typename CellIndex::offset_t offsetp = {};
    offsetm[dir] = -1;
    offsetp[dir] = 1;
    CellIndex iCell_Source_C = Source.convert_index(iCell_Uout);
    CellIndex iCell_Source_L = iCell_Source_C + offsetm;
    CellIndex iCell_Source_R = iCell_Source_C + offsetp;
    CellIndex iCell_Slopes_C = Slopes.convert_index(iCell_Uout);
    CellIndex iCell_Slopes_L = iCell_Slopes_C + offsetm;
    CellIndex iCell_Slopes_R = iCell_Slopes_C + offsetp;


    PrimState sourceL, sourceC, sourceR;
    PrimState slopeL, slopeC, slopeR;

    getPrimitiveState<ndim>(Source, iCell_Source_L, sourceL);
    getPrimitiveState<ndim>(Source, iCell_Source_C, sourceC);
    getPrimitiveState<ndim>(Source, iCell_Source_R, sourceR);

    getPrimitiveState<ndim>(Slopes, iCell_Slopes_L, slopeL);
    getPrimitiveState<ndim>(Slopes, iCell_Slopes_C, slopeC);
    getPrimitiveState<ndim>(Slopes, iCell_Slopes_R, slopeR);

    ConsState fluxL = compute_flux<ndim, State>( sourceL, sourceC, slopeL, slopeC, dir, smallr, params );
    ConsState fluxR = compute_flux<ndim, State>( sourceC, sourceR, slopeC, slopeR, dir, smallr, params );

    ConsState umod;
    getConservativeState<ndim>(Uout, iCell_Uout, umod);
    umod += (fluxL - fluxR) * dtddir;
    setConservativeState<ndim>(Uout, iCell_Uout, umod);
  }

  /**
   * @brief Applies corrector step for gravity
   * 
   * @tparam ndim the number of dimensions
   * @param Uin Initial values before update
   * @param iCell_Uin Position insides Uin/Uout (non ghosted)
   * @param dt time step
   * @param use_field Get gravity field from Uin
   * @param gx scalar values when use_field == false
   * @param gy
   * @param gz
   * @param Uout Updated array after hydro without gravity
   **/
  template<int ndim>
  KOKKOS_INLINE_FUNCTION
  void apply_gravity_correction( const GlobalArray& Uin,
                                 const CellIndex& iCell_Uin,
                                 real_t dt,
                                 bool use_field,
                                 real_t gx, real_t gy, real_t gz,
                                 const GlobalArray& Uout ){
    if(use_field)
    {
      gx = Uin.at(iCell_Uin, IGX);
      gy = Uin.at(iCell_Uin, IGY);
      if (ndim == 3)
        gz = Uin.at(iCell_Uin, IGZ);
    }

    real_t rhoOld = Uin.at(iCell_Uin, ID);
    
    real_t rhoNew = Uout.at(iCell_Uin, ID);
    real_t rhou = Uout.at(iCell_Uin, IU);
    real_t rhov = Uout.at(iCell_Uin, IV);
    real_t ekin_old = rhou*rhou + rhov*rhov;
    real_t rhow;
    
    if (ndim == 3) {
      rhow = Uout.at(iCell_Uin, IW);
      ekin_old += rhow*rhow;
    }
    
    ekin_old = 0.5 * ekin_old / rhoNew;

    rhou += 0.5 * dt * gx * (rhoOld + rhoNew);
    rhov += 0.5 * dt * gy * (rhoOld + rhoNew);

    Uout.at(iCell_Uin, IU) = rhou;
    Uout.at(iCell_Uin, IV) = rhov;
    if (ndim == 3) {
      rhow += 0.5 * dt * gz * (rhoOld + rhoNew);
      Uout.at(iCell_Uin, IW) = rhow;
    }

    // Energy correction should be included in case of self-gravitation ?
    real_t ekin_new = rhou*rhou + rhov*rhov;
    if (ndim == 3)
      ekin_new += rhow*rhow;
    
    ekin_new = 0.5 * ekin_new / rhoNew;
    Uout.at(iCell_Uin, IE) += (ekin_new - ekin_old);
  }    
} // namespace
} // namespace dyablo

#include "hydro/CopyGhostBlockCellData.h"

namespace dyablo {

template<typename State_>
class HydroUpdate_hancock : public HydroUpdate {
public:
  using State = State_;
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  HydroUpdate_hancock(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers )
  : foreach_cell(foreach_cell),
    xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
    ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
    zmin(configMap.getValue<real_t>("mesh", "zmin", 0.0)),
    xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
    ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
    zmax(configMap.getValue<real_t>("mesh", "zmax", 1.0)),
    timers(timers),
    params(configMap),
    boundary_type_xmin(configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING)),
    boundary_type_ymin(configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING)),
    boundary_type_zmin(configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)),
    gravity_type(configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE))
  {
    if (gravity_type & GRAVITY_CONSTANT) {
      gx = configMap.getValue<real_t>("gravity", "gx", 0.0);
      gy = configMap.getValue<real_t>("gravity", "gy", 0.0);
      gz = configMap.getValue<real_t>("gravity", "gz", 0.0);
    } 
  }

  ~HydroUpdate_hancock() {}

  /**
   * @brief Solves hydro for one step using the Muscl-Hancock method
   * 
   * @param Uin the input global array
   * @param Uout the output global array
   * @param dt the timestep
   */
  void update(
    const ForeachCell::CellArray_global_ghosted& Uin,
    const ForeachCell::CellArray_global_ghosted& Uout,
    real_t dt)
  {
    uint32_t ndim = foreach_cell.getDim();
    if(ndim == 2)
      update_aux<2>(Uin, Uout, dt);
    else
      update_aux<3>(Uin, Uout, dt);
  }

  template<int ndim>
  void update_aux(
      const ForeachCell::CellArray_global_ghosted& Uin,
      const ForeachCell::CellArray_global_ghosted& Uout,
      real_t dt)
  {
    const RiemannParams& params = this->params;  
    const int slope_type = this->slope_type;
    const real_t gamma = params.gamma0;
    const double smallr = params.smallr;
    const GravityType gravity_type = this->gravity_type;

    real_t xmin = this->xmin, ymin = this->ymin, zmin = this->zmin;
    real_t xmax = this->xmax, ymax = this->ymax, zmax = this->zmax;
    BoundaryConditionType xbound = this->boundary_type_xmin;
    BoundaryConditionType ybound = this->boundary_type_ymin;
    BoundaryConditionType zbound = this->boundary_type_zmin;

    bool has_gravity = gravity_type!=GRAVITY_NONE;
    bool gravity_use_field = gravity_type&GRAVITY_FIELD;
    #ifndef NDEBUG
    bool gravity_use_scalar = gravity_type==GRAVITY_CST_SCALAR;
    #endif
    real_t gx = this->gx, gy = this->gy, gz = this->gz;

    // If gravity is on it must either use the force field from U or a constant scalar force field
    assert( !has_gravity || ( gravity_use_field != gravity_use_scalar )  );

    Timers& timers = this->timers; 

    ForeachCell& foreach_cell = this->foreach_cell;

    auto fm = Uin.fm;
    
    // Create abstract temporary ghosted arrays for patches 
    PatchArray::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, fm, State::N);
    PatchArray::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, fm, State::N);
    PatchArray::Ref SlopesX_ = foreach_cell.reserve_patch_tmp("SlopesX", 1, 0, 0, fm, State::N);
    PatchArray::Ref SlopesY_ = foreach_cell.reserve_patch_tmp("SlopesY", 0, 1, 0, fm, State::N);
    PatchArray::Ref SlopesZ_;
    if( ndim == 3 )
      SlopesZ_ = foreach_cell.reserve_patch_tmp("SlopesZ", 0, 0, 1, fm, State::N);
    PatchArray::Ref Sources_ = foreach_cell.reserve_patch_tmp("Sources", 1, 1, (ndim == 3)?1:0, fm, State::N);

    timers.get("HydroUpdate_hancock").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Iterate over patches
    foreach_cell.foreach_patch( "HydroUpdate_hancock::update",
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      PatchArray Ugroup = patch.allocate_tmp(Ugroup_);
      PatchArray Qgroup = patch.allocate_tmp(Qgroup_);
      PatchArray SlopesX = patch.allocate_tmp(SlopesX_);
      PatchArray SlopesY = patch.allocate_tmp(SlopesY_);
      PatchArray SlopesZ = patch.allocate_tmp(SlopesZ_);
      PatchArray Sources = patch.allocate_tmp(Sources_);

      // Copy non ghosted array Uin into temporary ghosted Ugroup with two ghosts
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      {
          copyGhostBlockCellData<ndim>(
          Uin, iCell_Ugroup, 
          cellmetadata, 
          xmin, ymin, zmin, 
          xmax, ymax, zmax, 
          xbound, ybound, zbound,
          Ugroup);
      });

      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      { 
        compute_primitives<ndim, State>(params, Ugroup, iCell_Ugroup, Qgroup);
      });

      patch.foreach_cell(Sources, CELL_LAMBDA(const CellIndex& iCell_Sources)
      { 
        auto size = cellmetadata.getCellSize(iCell_Sources);
        compute_slopes<ndim, State>(
          Qgroup, iCell_Sources, slope_type, gamma, size[IX], size[IY], size[IZ], dt,
          Sources, SlopesX, SlopesY, SlopesZ
        );
      });
      
      patch.foreach_cell( Uout, CELL_LAMBDA(const CellIndex& iCell_Uout)
      {
        auto size = cellmetadata.getCellSize(iCell_Uout);
        ConsState u0;
        getConservativeState<ndim>(Uin, iCell_Uout, u0);
        setConservativeState<ndim>(Uout, iCell_Uout, u0);
        compute_fluxes<ndim, State>(IX, iCell_Uout, SlopesX, Sources, params, smallr, dt/size[IX], Uout);
        compute_fluxes<ndim, State>(IY, iCell_Uout, SlopesY, Sources, params, smallr, dt/size[IY], Uout);
        if( ndim==3 ) compute_fluxes<ndim, State>(IZ, iCell_Uout, SlopesZ, Sources, params, smallr, dt/size[IZ], Uout);
      
        // Applying correction step for gravity
        if (has_gravity)
          apply_gravity_correction<ndim>(Uin, iCell_Uout, dt, gravity_use_field, gx, gy, gz, Uout);
      });
    });

    timers.get("HydroUpdate_hancock").stop();

  }

private:
  ForeachCell& foreach_cell;
  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;  
  
  Timers& timers;  

  RiemannParams params;

  real_t slope_type;

  BoundaryConditionType boundary_type_xmin,boundary_type_ymin, boundary_type_zmin;
  GravityType gravity_type;
  real_t gx, gy, gz;
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::HydroUpdateFactory, 
                 dyablo::HydroUpdate_hancock<dyablo::HydroState>, 
                 "HydroUpdate_hancock")

FACTORY_REGISTER(dyablo::HydroUpdateFactory, 
                 dyablo::HydroUpdate_hancock<dyablo::MHDState>, 
                 "MHDUpdate_hancock")
