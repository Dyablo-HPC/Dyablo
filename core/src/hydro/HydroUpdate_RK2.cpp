#include "HydroUpdate_RK2.h"

#include "RiemannSolvers.h"
#include "HydroUpdate_utils.h"

namespace dyablo {

struct HydroUpdate_RK2::Data{ 
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


HydroUpdate_RK2::HydroUpdate_RK2(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers) 
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

HydroUpdate_RK2::~HydroUpdate_RK2() {
}

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
void rk_update(std::string kernel_name, 
               const HydroUpdate_RK2::Data* pdata, 
               const ForeachCell::CellArray_global_ghosted Uin, 
               const ForeachCell::CellArray_global_ghosted Uout, 
               real_t dt) 
{
  auto fm = Uin.fm;
  auto foreach_cell = pdata->foreach_cell;
  auto gravity_type = pdata->gravity_type;
  auto params = pdata->params;

  // Create abstract temporary ghosted arrays for patches 
  PatchArray::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, fm, 5);
  PatchArray::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, fm, 5);
  
  ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

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

  // Iterate over patches
  foreach_cell.foreach_patch( "HydroUpdate_euler::update",
    PATCH_LAMBDA( const ForeachCell::Patch& patch )
  {
    PatchArray Ugroup = patch.allocate_tmp(Ugroup_);
    PatchArray Qgroup = patch.allocate_tmp(Qgroup_);

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
    
    patch.foreach_cell( Uout, CELL_LAMBDA(const CellIndex& iCell_Uout)
    {
      auto size = cellmetadata.getCellSize(iCell_Uout);
      HydroState3d u0 = getHydroState<ndim>( Uin, iCell_Uout );
      setHydroState<ndim>( Uout, iCell_Uout, u0);
      euler_update<ndim>(params, IX, iCell_Uout, Uin, Qgroup, dt, size[IX], Uout);
      euler_update<ndim>(params, IY, iCell_Uout, Uin, Qgroup, dt, size[IY], Uout);
      if(ndim==3) 
        euler_update<ndim>(params, IZ, iCell_Uout, Uin, Qgroup, dt, size[IZ], Uout);
    
      // Applying correction step for gravity
      if (has_gravity)
        apply_gravity_correction<ndim>(Uin, iCell_Uout, dt, gravity_use_field, gx, gy, gz, Uout);
    });
  });
}

template< int ndim >
void rk_correct(const HydroUpdate_RK2::Data* pdata, 
                const ForeachCell::CellArray_global_ghosted Uin, 
                const ForeachCell::CellArray_global_ghosted Uout) 
{
  auto foreach_cell = pdata->foreach_cell;

  foreach_cell.foreach_patch("HydroUpdate_RK2::correct",
    PATCH_LAMBDA( const ForeachCell::Patch& patch )
  {
    patch.foreach_cell( Uout, CELL_LAMBDA(const CellIndex &iCell_Uout) {
      auto iCell_Uin = Uin.convert_index(iCell_Uout);
      
      const HydroState3d uin  = getHydroState<ndim>(Uin,  iCell_Uin);
      const HydroState3d uout = getHydroState<ndim>(Uout, iCell_Uout);
      const HydroState3d res = 0.5 * (uin + uout);
      
      setHydroState<ndim>( Uout, iCell_Uout, res);
      });
  });
}

template< int ndim >
void update_aux(
    const HydroUpdate_RK2::Data* pdata,
    const ForeachCell::CellArray_global_ghosted& Uin,
    const ForeachCell::CellArray_global_ghosted& Uout,
    real_t dt)
{
  Timers& timers = pdata->timers; 
  ForeachCell& foreach_cell = pdata->foreach_cell;
  GhostCommunicator ghost_comm(std::shared_ptr<AMRmesh>(&foreach_cell.get_amr_mesh(), [](AMRmesh*){}));
    

  // Temporary Ustar array
  auto fm = FieldManager(Uin.fm.enabled_fields()); 
  auto Ustar = foreach_cell.allocate_ghosted_array("U*", fm);
  
  // Two update stages and one correction stage
  timers.get("HydroUpdate_RK2").start();
  rk_update<ndim>("update1", pdata, Uin,  Ustar, dt);
  Ustar.exchange_ghosts(ghost_comm);
  rk_update<ndim>("update2", pdata, Ustar, Uout, dt);
  rk_correct<ndim>(pdata, Uin, Uout);

  timers.get("HydroUpdate_RK2").stop();
}
}

void HydroUpdate_RK2::update(
    const ForeachCell::CellArray_global_ghosted& Uin,
    const ForeachCell::CellArray_global_ghosted& Uout,
    real_t dt)
{
  uint32_t ndim = pdata->foreach_cell.getDim();
  if(ndim == 2)
    update_aux<2>(this->pdata.get(), Uin, Uout, dt);
  else
    update_aux<3>(this->pdata.get(), Uin, Uout, dt);  
}
}

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_RK2, 
                  "HydroUpdate_RK2")
