#include "HydroUpdate_generic.h"

#include "utils/monitoring/Timers.h"

#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "RiemannSolvers.h"


namespace dyablo { 


struct HydroUpdate_generic::Data{ 
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

HydroUpdate_generic::HydroUpdate_generic(
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

HydroUpdate_generic::~HydroUpdate_generic()
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

template< int ndim >
KOKKOS_INLINE_FUNCTION
void compute_primitives(const RiemannParams& params, const PatchArray& Ugroup, const CellIndex& iCell_Ugroup, const PatchArray& Qgroup)
{
  HydroState3d uLoc = getHydroState<ndim>( Ugroup, iCell_Ugroup );
      
  // get primitive variables in current cell
  HydroState3d qLoc;
  real_t c = 0.0;
  if(ndim==3)
    computePrimitives(uLoc, &c, qLoc, params.gamma0, params.smallr, params.smallp);
  else
  {
    auto copy_state = [](auto& to, const auto& from){
      to[ID] = from[ID];
      to[IP] = from[IP];
      to[IU] = from[IU];
      to[IV] = from[IV];
    };
    HydroState2d uLoc_2d, qLoc_2d;
    copy_state(uLoc_2d, uLoc);
    computePrimitives(uLoc_2d, &c, qLoc_2d, params.gamma0, params.smallr, params.smallp);
    copy_state(qLoc, qLoc_2d);
  }

  setHydroState<ndim>( Qgroup, iCell_Ugroup, qLoc );
}

/// Compute slope for one cell (q_) in one direction
template< int ndim >
KOKKOS_INLINE_FUNCTION
HydroState3d compute_slope( const HydroState3d& qMinus_, 
                            const HydroState3d& q_, 
                            const HydroState3d& qPlus_, 
                            int slope_type)
{
  assert(slope_type == 1 || slope_type == 2);

  HydroState3d dq_ = {};
  auto compute_slope_var = [&](VarIndex ivar)
  {
    const real_t& q = q_[ivar];
    const real_t& qPlus = qPlus_[ivar];
    const real_t& qMinus = qMinus_[ivar];
    real_t& dq = dq_[ivar];

    // slopes in first coordinate direction
    const real_t dlft = slope_type * (q - qMinus);
    const real_t drgt = slope_type * (qPlus - q);
    const real_t dcen = HALF_F * (qPlus - qMinus);
    const real_t dsgn = (dcen >= ZERO_F) ? ONE_F : -ONE_F;
    if( std::isnan(dlft) or std::isnan(drgt)  )
    {
      dq_[ivar] = std::nan("");
      return;
    }
    const real_t slop = fmin(FABS(dlft), FABS(drgt));
    real_t dlim = slop;
    if ((dlft * drgt) <= ZERO_F)
      dlim = ZERO_F;
    dq = dsgn * fmin(dlim, FABS(dcen));
  };

  compute_slope_var(ID);
  compute_slope_var(IP);
  compute_slope_var(IU);
  compute_slope_var(IV);
  if(ndim==3) compute_slope_var(IW);

  return dq_;
}

template< int ndim >
KOKKOS_INLINE_FUNCTION
HydroState3d compute_source(  const HydroState3d& q,
                              const HydroState3d& slopeX,
                              const HydroState3d& slopeY,
                              const HydroState3d& slopeZ,
                              real_t dtdx, real_t dtdy, real_t dtdz,
                              real_t gamma )
{
  // retrieve primitive variables in current quadrant
  const real_t r = q[ID];
  const real_t p = q[IP];
  const real_t u = q[IU];
  const real_t v = q[IV];
  const real_t w = q[IW];

  // retrieve variations = dx * slopes
  const real_t drx = slopeX[ID] * 0.5;
  const real_t dpx = slopeX[IP] * 0.5;
  const real_t dux = slopeX[IU] * 0.5;
  const real_t dvx = slopeX[IV] * 0.5;
  const real_t dwx = slopeX[IW] * 0.5;    
  const real_t dry = slopeY[ID] * 0.5;
  const real_t dpy = slopeY[IP] * 0.5;
  const real_t duy = slopeY[IU] * 0.5;
  const real_t dvy = slopeY[IV] * 0.5;
  const real_t dwy = slopeY[IW] * 0.5;    
  const real_t drz = slopeZ[ID] * 0.5;
  const real_t dpz = slopeZ[IP] * 0.5;
  const real_t duz = slopeZ[IU] * 0.5;
  const real_t dvz = slopeZ[IV] * 0.5;
  const real_t dwz = slopeZ[IW] * 0.5;

  HydroState3d source;
  if( ndim == 3 )
  {
    source[ID] = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy + (-w * drz - dwz * r) * dtdz;
    source[IU] = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy + (-w * duz) * dtdz;
    source[IV] = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy + (-w * dvz) * dtdz;
    source[IW] = w + (-u * dwx) * dtdx + (-v * dwy) * dtdy + (-w * dwz - dpz / r) * dtdz;
    source[IP] = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy + (-w * dpz - dwz * gamma * p) * dtdz;
  }
  else
  {
    source[ID] = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
    source[IU] = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
    source[IV] = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
    source[IP] = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;
  }
  return source;
}

template< int ndim >
KOKKOS_INLINE_FUNCTION
void compute_slopes( const PatchArray& Qgroup, const CellIndex& iCell_Sources, int slope_type, real_t gamma, real_t dx, real_t dy, real_t dz, real_t dt,
                     const PatchArray& Sources, const PatchArray& SlopesX, const PatchArray& SlopesY, const PatchArray& SlopesZ)
{
  using o_t = typename CellIndex::offset_t;

  HydroState3d sx, sy, sz;

  CellIndex ib = Qgroup.convert_index(iCell_Sources);
  HydroState3d qc = getHydroState<ndim>( Qgroup, ib );

  {
    // neighbor along x axis
    HydroState3d qm = getHydroState<ndim>( Qgroup, ib + o_t{-1, 0, 0});
    HydroState3d qp = getHydroState<ndim>( Qgroup, ib + o_t{ 1, 0, 0});     

    sx = compute_slope<ndim>(qm, qc, qp, slope_type);
    CellIndex iCell_x = SlopesX.convert_index_ghost(iCell_Sources);
    if(iCell_x.is_valid())
    {
      setHydroState<ndim>(SlopesX, iCell_x, sx);
    }
  }

  {
    // neighbor along y axis
    HydroState3d qm = getHydroState<ndim>( Qgroup, ib + o_t{ 0,-1, 0});
    HydroState3d qp = getHydroState<ndim>( Qgroup, ib + o_t{ 0, 1, 0});       

    sy = compute_slope<ndim>(qm, qc, qp, slope_type);

    CellIndex iCell_y = SlopesY.convert_index_ghost(iCell_Sources);
    if(iCell_y.is_valid())
    {
      setHydroState<ndim>(SlopesY, iCell_y, sy);
    }
  }

  if( ndim == 3 )
  {      
    // neighbor along z axis
    HydroState3d qm = getHydroState<ndim>( Qgroup, ib + o_t{ 0, 0,-1});
    HydroState3d qp = getHydroState<ndim>( Qgroup, ib + o_t{ 0, 0, 1});       

    sz = compute_slope<ndim>(qm, qc, qp, slope_type);

    CellIndex iCell_z = SlopesZ.convert_index_ghost(iCell_Sources);
    if(iCell_z.is_valid())
    {
      setHydroState<ndim>(SlopesZ, iCell_z, sz);
    }
  }

  {
    HydroState3d source = compute_source<ndim>(qc, sx, sy, sz, dt/dx, dt/dy, dt/dz, gamma);
    setHydroState<ndim>(Sources, iCell_Sources, source);
  }
}

KOKKOS_INLINE_FUNCTION
HydroState3d reconstruct_state( const HydroState3d& source, 
                                const HydroState3d& slope,
                                real_t sign, real_t smallr )
{
  HydroState3d res;

  res[ID] = source[ID] + sign * slope[ID] * 0.5;
  res[IP] = source[IP] + sign * slope[IP] * 0.5;
  res[IU] = source[IU] + sign * slope[IU] * 0.5;
  res[IV] = source[IV] + sign * slope[IV] * 0.5;
  res[IW] = source[IW] + sign * slope[IW] * 0.5;

  res[ID] = fmax(smallr, res[ID]);

  return res;
}

template< int ndim >
/// Reconstruct state and apply Riemann solver at the interface between cell L and R
KOKKOS_INLINE_FUNCTION
HydroState3d compute_flux( const HydroState3d& sourceL, const HydroState3d& sourceR, 
                           const HydroState3d& slopeL, const HydroState3d& slopeR,
                           ComponentIndex3D dir, real_t smallr, const RiemannParams& params )
{
  HydroState3d qL = reconstruct_state( sourceL, slopeL, 1, smallr );
  HydroState3d qR = reconstruct_state( sourceR, slopeR, -1, smallr );

  VarIndex swap_component = (dir==IX) ? IU : (dir==IY) ? IV : IW;

  // riemann solver along Y or Z direction requires to 
  // swap velocity components
  swap(qL[IU], qL[swap_component]);
  swap(qR[IU], qR[swap_component]);

  // step 4 : compute flux (Riemann solver)
  HydroState3d flux;
  if( ndim == 3 )
    flux = riemann_hydro(qL, qR, params);
  else
  {
    auto copy_state = [](auto& to, const auto& from){
      to[ID] = from[ID];
      to[IP] = from[IP];
      to[IU] = from[IU];
      to[IV] = from[IV];
    };
    HydroState2d qL_2d, qR_2d;
    copy_state(qL_2d, qL);
    copy_state(qR_2d, qR);
    HydroState2d flux_2d = riemann_hydro(qL_2d, qR_2d, params);
    copy_state(flux, flux_2d);    
  }

  swap(flux[IU], flux[swap_component]);

  return flux;
}

/**
 * Compute both fluxes (using Riemann solver) in a direction for one cell and update U2
 * @note this version is non-conservative because non-conforming interfaces are not taken into account
 **/
template< int ndim >
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


  HydroState3d sourceL = getHydroState<ndim>( Source, iCell_Source_L );
  HydroState3d sourceC = getHydroState<ndim>( Source, iCell_Source_C );
  HydroState3d sourceR = getHydroState<ndim>( Source, iCell_Source_R );
  HydroState3d slopeL  = getHydroState<ndim>( Slopes, iCell_Slopes_L );
  HydroState3d slopeC  = getHydroState<ndim>( Slopes, iCell_Slopes_C );
  HydroState3d slopeR  = getHydroState<ndim>( Slopes, iCell_Slopes_R );

  HydroState3d fluxL = compute_flux<ndim>( sourceL, sourceC, slopeL, slopeC, dir, smallr, params );
  HydroState3d fluxR = compute_flux<ndim>( sourceC, sourceR, slopeC, slopeR, dir, smallr, params );

  Uout.at(iCell_Uout, ID) += (fluxL[ID] - fluxR[ID]) * dtddir;
  Uout.at(iCell_Uout, IP) += (fluxL[IP] - fluxR[IP]) * dtddir;
  Uout.at(iCell_Uout, IU) += (fluxL[IU] - fluxR[IU]) * dtddir;
  Uout.at(iCell_Uout, IV) += (fluxL[IV] - fluxR[IV]) * dtddir;
  if(ndim==3) Uout.at(iCell_Uout, IW) += (fluxL[IW] - fluxR[IW]) * dtddir;
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

template< int ndim >
void update_aux(
    const HydroUpdate_generic::Data* pdata,
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

  timers.get("HydroUpdate_generic").start();

  ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

  // Iterate over patches
  foreach_cell.foreach_patch( "HydroUpdate_generic::update",
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
      compute_primitives<ndim>(params, Ugroup, iCell_Ugroup, Qgroup);
    });

    patch.foreach_cell(Sources, CELL_LAMBDA(const CellIndex& iCell_Sources)
    { 
      auto size = cellmetadata.getCellSize(iCell_Sources);
      compute_slopes<ndim>(
        Qgroup, iCell_Sources, slope_type, gamma, size[IX], size[IY], size[IZ], dt,
        Sources, SlopesX, SlopesY, SlopesZ
      );
    });
    
    patch.foreach_cell( Uout, CELL_LAMBDA(const CellIndex& iCell_Uout)
    {
      auto size = cellmetadata.getCellSize(iCell_Uout);
      HydroState3d u0 = getHydroState<ndim>( Uin, iCell_Uout );
      setHydroState<ndim>( Uout, iCell_Uout, u0);
      compute_fluxes<ndim>(IX, iCell_Uout, SlopesX, Sources, params, smallr, dt/size[IX], Uout);
      compute_fluxes<ndim>(IY, iCell_Uout, SlopesY, Sources, params, smallr, dt/size[IY], Uout);
      if( ndim==3 ) compute_fluxes<ndim>(IZ, iCell_Uout, SlopesZ, Sources, params, smallr, dt/size[IZ], Uout);
    
      // Applying correction step for gravity
      if (has_gravity)
        apply_gravity_correction<ndim>(Uin, iCell_Uout, dt, gravity_use_field, gx, gy, gz, Uout);
    });
  });

  timers.get("HydroUpdate_generic").stop();

}

} // namespace

void HydroUpdate_generic::update(
    const ForeachCell::CellArray_global_ghosted& Uin,
    const ForeachCell::CellArray_global_ghosted& Uout,
    real_t dt)
{
  uint32_t ndim = pdata->foreach_cell.getDim();
  if(ndim == 2)
    update_aux<2>( this->pdata.get(), Uin, Uout, dt );
  else
    update_aux<3>( this->pdata.get(), Uin, Uout, dt );
}

}// namespace dyablo


FACTORY_REGISTER( dyablo::HydroUpdateFactory , dyablo::HydroUpdate_generic, "HydroUpdate_generic")
