#include "HydroUpdate_hancock.h"

#include "utils/monitoring/Timers.h"

#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "RiemannSolvers.h"

#include "HydroUpdate_utils.h"


namespace dyablo { 


struct HydroUpdate_hancock::Data{ 
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

HydroUpdate_hancock::HydroUpdate_hancock(
  ConfigMap& configMap,
  ForeachCell& foreach_cell,
  Timers& timers )
 : pdata(new Data
    {foreach_cell,
    configMap.getValue<real_t>("mesh", "xmin", 0.0),
    configMap.getValue<real_t>("mesh", "ymin", 0.0),
    configMap.getValue<real_t>("mesh", "zmin", 0.0),
    configMap.getValue<real_t>("mesh", "xmax", 1.0),
    configMap.getValue<real_t>("mesh", "ymax", 1.0),
    configMap.getValue<real_t>("mesh", "zmax", 1.0),
    timers,
    RiemannParams( configMap ),
    configMap.getValue<real_t>("hydro","slope_type",1.0),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING),
    configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE)
    })
{
  if (pdata->gravity_type & GRAVITY_CONSTANT) {
    pdata->gx = configMap.getValue<real_t>("gravity", "gx",  0.0);
    pdata->gy = configMap.getValue<real_t>("gravity", "gy",  0.0);
    pdata->gz = configMap.getValue<real_t>("gravity", "gz",  0.0);
  } 
}

HydroUpdate_hancock::~HydroUpdate_hancock()
{}

namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray = ForeachCell::CellArray_global;
using PatchArray = ForeachCell::CellArray_patch;
using CellIndex = ForeachCell::CellIndex;

}// namespace
}// namespace dyablo


#include "hydro/CopyGhostBlockCellData.h"

namespace dyablo { 

namespace{


template < 
  int ndim,
  typename PrimState,
  typename ConsState >
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

template < 
  int ndim,
  typename PrimState,
  typename ConsState >
KOKKOS_INLINE_FUNCTION
void compute_slopes( const PatchArray& Qgroup, const CellIndex& iCell_Sources, int slope_type, real_t gamma, real_t dx, real_t dy, real_t dz, real_t dt,
                     const PatchArray& Sources, const PatchArray& SlopesX, const PatchArray& SlopesY, const PatchArray& SlopesZ)
{
  using o_t = typename CellIndex::offset_t;

  PrimState sx{}, sy{}, sz{};

  CellIndex ib = Qgroup.convert_index(iCell_Sources);
  PrimState qc = getPrimitiveState<ndim>( Qgroup, ib );

  {
    // neighbor along x axis
    PrimState qm = getPrimitiveState<ndim>( Qgroup, ib + o_t{-1, 0, 0});
    PrimState qp = getPrimitiveState<ndim>( Qgroup, ib + o_t{ 1, 0, 0});     

    sx = compute_slope<ndim>(qm, qc, qp, 1.0, 1.0);
    CellIndex iCell_x = SlopesX.convert_index_ghost(iCell_Sources);
    if(iCell_x.is_valid())
      setPrimitiveState<ndim>(SlopesX, iCell_x, sx);
  }

  {
    // neighbor along y axis
    PrimState qm = getPrimitiveState<ndim>( Qgroup, ib + o_t{ 0,-1, 0});
    PrimState qp = getPrimitiveState<ndim>( Qgroup, ib + o_t{ 0, 1, 0});       

    sy = compute_slope<ndim>(qm, qc, qp, 1.0, 1.0);

    CellIndex iCell_y = SlopesY.convert_index_ghost(iCell_Sources);
    if(iCell_y.is_valid())
      setPrimitiveState<ndim>(SlopesY, iCell_y, sy);
  }

  if( ndim == 3 )
  {      
    // neighbor along z axis
    PrimState qm = getPrimitiveState<ndim>( Qgroup, ib + o_t{ 0, 0,-1});
    PrimState qp = getPrimitiveState<ndim>( Qgroup, ib + o_t{ 0, 0, 1});       

    sz = compute_slope<ndim>(qm, qc, qp, 1.0, 1.0);

    CellIndex iCell_z = SlopesZ.convert_index_ghost(iCell_Sources);
    if(iCell_z.is_valid())
      setPrimitiveState<ndim>(SlopesZ, iCell_z, sz);
  }

  {
    PrimState source = compute_source<ndim, PrimState, ConsState>(qc, sx, sy, sz, dt/dx, dt/dy, dt/dz, gamma);
    setPrimitiveState<ndim>(Sources, iCell_Sources, source);
  }
}

template < typename PrimState >
KOKKOS_INLINE_FUNCTION
PrimState reconstruct_state(const PrimState& source, 
                            const PrimState& slope,
                            real_t sign, real_t smallr )
{
  PrimState res;

  res.rho = source.rho + sign * slope.rho * 0.5;
  res.p   = source.p   + sign * slope.p * 0.5;
  res.u   = source.u   + sign * slope.u * 0.5;
  res.v   = source.v   + sign * slope.v * 0.5;
  res.w   = source.w   + sign * slope.w * 0.5;

  res.rho = fmax(smallr, res.rho);

  return res;
}

template< 
  int ndim, 
  typename PrimState,
  typename ConsState >
/// Reconstruct state and apply Riemann solver at the interface between cell L and R
KOKKOS_INLINE_FUNCTION
ConsState compute_flux( const PrimState& sourceL, const PrimState& sourceR, 
                        const PrimState& slopeL,  const PrimState& slopeR,
                        ComponentIndex3D dir, real_t smallr, const RiemannParams& params )
{
  PrimState qL = reconstruct_state( sourceL, slopeL, 1, smallr );
  PrimState qR = reconstruct_state( sourceR, slopeR, -1, smallr );

  // riemann solver along Y or Z direction requires to 
  // swap velocity components
  if (dir == IY) {
    swap(qL.u, qL.v);
    swap(qR.u, qR.v);
  }
  else if (dir == IZ) {
    swap(qL.u, qL.w);
    swap(qR.u, qR.w);
  }

  // step 4 : compute flux (Riemann solver)
  ConsState flux = riemann_hydro(qL, qR, params);
  
  if (dir == IY)
    swap(flux.rho_u, flux.rho_v);
  else if (dir == IZ)
    swap(flux.rho_u, flux.rho_w);

  return flux;
}

/**
 * Compute both fluxes (using Riemann solver) in a direction for one cell and update U2
 * @note this version is non-conservative because non-conforming interfaces are not taken into account
 **/
template < 
  int ndim,
  typename PrimState,
  typename ConsState>
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


  PrimState sourceL = getPrimitiveState<ndim>( Source, iCell_Source_L );
  PrimState sourceC = getPrimitiveState<ndim>( Source, iCell_Source_C );
  PrimState sourceR = getPrimitiveState<ndim>( Source, iCell_Source_R );
  PrimState slopeL  = getPrimitiveState<ndim>( Slopes, iCell_Slopes_L );
  PrimState slopeC  = getPrimitiveState<ndim>( Slopes, iCell_Slopes_C );
  PrimState slopeR  = getPrimitiveState<ndim>( Slopes, iCell_Slopes_R );

  ConsState fluxL = compute_flux<ndim, PrimState, ConsState>( sourceL, sourceC, slopeL, slopeC, dir, smallr, params );
  ConsState fluxR = compute_flux<ndim, PrimState, ConsState>( sourceC, sourceR, slopeC, slopeR, dir, smallr, params );

  Uout.at(iCell_Uout, ID) += (fluxL.rho - fluxR.rho) * dtddir;
  Uout.at(iCell_Uout, IE) += (fluxL.e_tot - fluxR.e_tot) * dtddir;
  Uout.at(iCell_Uout, IU) += (fluxL.rho_u - fluxR.rho_u) * dtddir;
  Uout.at(iCell_Uout, IV) += (fluxL.rho_v - fluxR.rho_v) * dtddir;
  if(ndim==3) Uout.at(iCell_Uout, IW) += (fluxL.rho_w - fluxR.rho_w) * dtddir;
}

/**
 * Applies corrector step for gravity
 * @param Uin Initial values before update
 * @param iCell_Uin Position insides Uin/Uout (non ghosted)
 * @param dt time step
 * @param use_field Get gravity field from Uin
 * @param gx, gy, gz, scalar values when use_field == false
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

template< 
  int ndim,
  typename PrimState,
  typename ConsState >
void update_aux(
    const HydroUpdate_hancock::Data* pdata,
    const ForeachCell::CellArray_global_ghosted& Uin,
    const ForeachCell::CellArray_global_ghosted& Uout,
    real_t dt)
{
  
  const RiemannParams& params = pdata->params;  
  const int slope_type = pdata->slope_type;
  const real_t gamma = params.gamma0;
  const double smallr = params.smallr;
  const GravityType gravity_type = pdata->gravity_type;

  real_t xmin = pdata->xmin, ymin = pdata->ymin, zmin = pdata->zmin;
  real_t xmax = pdata->xmax, ymax = pdata->ymax, zmax = pdata->zmax;
  BoundaryConditionType xbound = pdata->boundary_type_xmin;
  BoundaryConditionType ybound = pdata->boundary_type_ymin;
  BoundaryConditionType zbound = pdata->boundary_type_zmin;

  bool has_gravity = gravity_type!=GRAVITY_NONE;
  bool gravity_use_field = gravity_type&GRAVITY_FIELD;
  #ifndef NDEBUG
  bool gravity_use_scalar = gravity_type==GRAVITY_CST_SCALAR;
  #endif
  real_t gx = pdata->gx, gy = pdata->gy, gz = pdata->gz;

  // If gravity is on it must either use the force field from U or a constant scalar force field
  assert( !has_gravity || ( gravity_use_field != gravity_use_scalar )  );

  Timers& timers = pdata->timers; 

  ForeachCell& foreach_cell = pdata->foreach_cell;

  auto fm = Uin.fm;
  
  // Create abstract temporary ghosted arrays for patches 
  PatchArray::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, fm, 5);
  PatchArray::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, fm, 5);
  PatchArray::Ref SlopesX_ = foreach_cell.reserve_patch_tmp("SlopesX", 1, 0, 0, fm, 5);
  PatchArray::Ref SlopesY_ = foreach_cell.reserve_patch_tmp("SlopesY", 0, 1, 0, fm, 5);
  PatchArray::Ref SlopesZ_;
  if( ndim == 3 )
    SlopesZ_ = foreach_cell.reserve_patch_tmp("SlopesZ", 0, 0, 1, fm, 5);
  PatchArray::Ref Sources_ = foreach_cell.reserve_patch_tmp("Sources", 1, 1, (ndim == 3)?1:0, fm, 5);

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
      compute_primitives<ndim, PrimState, ConsState>(params, Ugroup, iCell_Ugroup, Qgroup);
    });

    patch.foreach_cell(Sources, CELL_LAMBDA(const CellIndex& iCell_Sources)
    { 
      auto size = cellmetadata.getCellSize(iCell_Sources);
      compute_slopes<ndim, PrimState, ConsState>(
        Qgroup, iCell_Sources, slope_type, gamma, size[IX], size[IY], size[IZ], dt,
        Sources, SlopesX, SlopesY, SlopesZ
      );
    });
    
    patch.foreach_cell( Uout, CELL_LAMBDA(const CellIndex& iCell_Uout)
    {
      auto size = cellmetadata.getCellSize(iCell_Uout);
      ConsHydroState u0 = getConservativeState<ndim>( Uin, iCell_Uout );
      setConservativeState<ndim>( Uout, iCell_Uout, u0);
      compute_fluxes<ndim, PrimState, ConsState>(IX, iCell_Uout, SlopesX, Sources, params, smallr, dt/size[IX], Uout);
      compute_fluxes<ndim, PrimState, ConsState>(IY, iCell_Uout, SlopesY, Sources, params, smallr, dt/size[IY], Uout);
      if( ndim==3 ) compute_fluxes<ndim, PrimState, ConsState>(IZ, iCell_Uout, SlopesZ, Sources, params, smallr, dt/size[IZ], Uout);
    
      // Applying correction step for gravity
      if (has_gravity)
        apply_gravity_correction<ndim>(Uin, iCell_Uout, dt, gravity_use_field, gx, gy, gz, Uout);
    });
  });

  timers.get("HydroUpdate_hancock").stop();

}

} // namespace

void HydroUpdate_hancock::update(
    const ForeachCell::CellArray_global_ghosted& Uin,
    const ForeachCell::CellArray_global_ghosted& Uout,
    real_t dt)
{
  uint32_t ndim = pdata->foreach_cell.getDim();
  if(ndim == 2)
    update_aux<2, PrimHydroState, ConsHydroState>( this->pdata.get(), Uin, Uout, dt );
  else
    update_aux<3, PrimHydroState, ConsHydroState>( this->pdata.get(), Uin, Uout, dt );
}

}// namespace dyablo


FACTORY_REGISTER( dyablo::HydroUpdateFactory , dyablo::HydroUpdate_hancock, "HydroUpdate_hancock")
