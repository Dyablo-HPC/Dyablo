#include "HydroUpdate_base.h"
#include "RiemannSolvers.h"
#include "HydroUpdate_utils.h"

namespace dyablo {
namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray = ForeachCell::CellArray_global;
using PatchArray = ForeachCell::CellArray_patch;
using CellIndex = ForeachCell::CellIndex;

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

}// namespace
}// namespace dyablo

#include "hydro/CopyGhostBlockCellData.h"

namespace dyablo {

/**
 * @brief Euler update algorithm
 * 
 * @tparam State the type of state to treat
 */
template<typename State_>
class HydroUpdate_euler : public HydroUpdate {
public:
  using State     = State_;
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  HydroUpdate_euler(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
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

  ~HydroUpdate_euler() {}

  /**
   * @brief Solves hydro for one step using the euler method
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
    if (ndim == 2)
      update_aux<2>(Uin, Uout, dt);
    else if (ndim == 3)
      update_aux<3>(Uin, Uout, dt);  
    else
      assert(false);
  }

  template< 
    int ndim >
  void update_aux(
      const ForeachCell::CellArray_global_ghosted& Uin,
      const ForeachCell::CellArray_global_ghosted& Uout,
      real_t dt) 
  {

    const RiemannParams& params = this->params; 
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
    
    timers.get("HydroUpdate_euler").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Iterate over patches
    foreach_cell.foreach_patch( "HydroUpdate_euler::update",
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      PatchArray Ugroup = patch.allocate_tmp(Ugroup_);
      PatchArray Qgroup = patch.allocate_tmp(Qgroup_);

      // Copy non ghosted array Uin into temporary ghosted Ugroup with two ghosts
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      {
        copyGhostBlockCellData<ndim, State>(
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
      
      patch.foreach_cell( Uout, CELL_LAMBDA(const CellIndex& iCell_Uout)
      {
        auto size = cellmetadata.getCellSize(iCell_Uout);
        ConsState u0{};
        getConservativeState<ndim>(Uin,  iCell_Uout, u0);
        setConservativeState<ndim>(Uout, iCell_Uout, u0);
        euler_update<ndim, State>(params, IX, iCell_Uout, Uin, Qgroup, dt, size[IX], Uout);
        euler_update<ndim, State>(params, IY, iCell_Uout, Uin, Qgroup, dt, size[IY], Uout);
        if(ndim==3)   
          euler_update<ndim, State>(params, IZ, iCell_Uout, Uin, Qgroup, dt, size[IZ], Uout);
      
        // Applying correction step for gravity
        if (has_gravity)
          apply_gravity_correction<ndim>(Uin, iCell_Uout, dt, gravity_use_field, gx, gy, gz, Uout);

        {
          ConsState u{};
          getConservativeState<ndim>(Uout, iCell_Uout, u);
          // This is fugly. Can't we do better ? We shouldn't need this for the code to be stable ...
          PrimState q = consToPrim<ndim>(u, params.gamma0);
          if (q.rho < 0.0) {
            printf("WARNING ! Negative density detected !!!\n");
            q.rho = params.smallr;
          }
          if (q.p < 0.0) {
            printf("WARNING ! Negative pressure detected !!!\n");
            q.p   = params.smallp;
          }
          u = primToCons<ndim>(q, params.gamma0);
          setConservativeState<ndim>(Uout, iCell_Uout, u);
        }
      });
    });

    timers.get("HydroUpdate_euler").stop();
  }

private:
  ForeachCell& foreach_cell;
  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;  
  
  Timers& timers;  

  RiemannParams params;

  BoundaryConditionType boundary_type_xmin,boundary_type_ymin, boundary_type_zmin;
  GravityType gravity_type;
  real_t gx, gy, gz;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_euler<dyablo::HydroState>, 
                  "HydroUpdate_euler")

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_euler<dyablo::MHDState>, 
                  "MHDUpdate_euler")
