#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "hydro/HydroUpdate_base.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "utils_hydro.h"
#include "RiemannSolvers.h"
#include "utils/config/ConfigMap.h"

#include "hydro/HydroUpdate_utils.h"

#include "states/State_forward.h"
#include "mpi/GhostCommunicator.h"

#include "boundary_conditions/BoundaryConditions.h"

#ifdef __CUDA_ARCH__
#include "math_constants.h"
#endif

class Timers;
class ConfigMap;

namespace dyablo {


namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using CellIndex = ForeachCell::CellIndex;
using FieldArray = UserData::FieldAccessor;

//Copied from HydroUpdate_generic
template < 
  int ndim,
  typename State>
KOKKOS_INLINE_FUNCTION
void computePrimitives(const RiemannParams& params, const FieldArray& Ugroup, 
                       const CellIndex& iCell_Ugroup, const GhostedArray& Qgroup)
{
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  ConsState uLoc{};
  getConservativeState<ndim>( Ugroup, iCell_Ugroup, uLoc );
  PrimState qLoc = consToPrim<ndim>( uLoc, params.gamma0 );
  setPrimitiveState<ndim>( Qgroup, iCell_Ugroup, qLoc );
}

KOKKOS_INLINE_FUNCTION
real_t big_real()
{
  return 1e99;
}

/**
 * Compute limited slopes and store result in Slopes_xyz
 * @tparam ndim 2d or 3d
 * @param[in] Q Array containing primitive variables
 * @param[in] iCell_U is current cell index
 * @param pos_cell cell center position
 * @param cell_size size of the cell
 * @param Slope_xyz limited slopes for every cell (in the xyz direction)
 **/
template< int ndim, typename State >
KOKKOS_INLINE_FUNCTION
void compute_limited_slopes(const GhostedArray& Q, const CellIndex& iCell_U, 
                            ForeachCell::CellMetaData::pos_t pos_cell, ForeachCell::CellMetaData::pos_t cell_size,
                            const GhostedArray& Slope_x, const GhostedArray& Slope_y, const GhostedArray& Slope_z)
{
  using PrimState = typename State::PrimState;

  PrimState grad[ndim];
  for (int d=0; d < ndim; ++d) 
    state_foreach_var( [](real_t& res){ res=big_real(); }, grad[d] );        

  PrimState qP{};
  getPrimitiveState<ndim>(Q, iCell_U, qP);
  PrimState q = qP;
  
  /// Compute gradient and apply slope limiter for neighbor 'iCell_neighbor' at position 'pos_neighbor' and in direction 'dir'
  auto update_minmod =[&]( const ComponentIndex3D& dir, const CellIndex& iCell_neighbor, real_t pos_n )
  {
    // default returned value (limited gradient didn't change)
    auto old_value = grad[dir];
    auto new_value = old_value;

    // compute distance along direction "dir" between current and
    // neighbor cell
    real_t pos_c = pos_cell[dir];
    real_t delta_x = pos_n - pos_c;

    PrimState qNeighP{};
    getPrimitiveState<ndim>(Q, iCell_neighbor, qNeighP);
    PrimState qNeigh = qNeighP;

    PrimState new_grad = (qNeigh - q)/delta_x;

    // this first test ensure a correct initialization
    state_foreach_var( 
      [](real_t& new_value, real_t old_value, real_t new_grad)
    {
      if ( old_value == big_real() )
        new_value = new_grad;
      else if (old_value * new_grad < 0)
        new_value = 0.0;
      else if ( fabs(new_grad) < fabs(old_value) )
        new_value = new_grad;
    }, new_value, old_value, new_grad ); 

    grad[dir] = new_value;
  };

  // Compute gradient and apply slope limiter for every neighbor in direction dir (sign = Left(-1)/Right(1))
  auto process_neighbors = [&]( ComponentIndex3D dir, int sign )
  {
      CellIndex::offset_t offset{};
      offset[dir] = sign;
      CellIndex iCell_n0 = iCell_U.getNeighbor_ghost( offset, Q );

      if( iCell_n0.is_boundary() )
        return;
      if( iCell_n0.level_diff() == 0 ) // same size
          update_minmod(dir, iCell_n0, pos_cell[dir]+offset[dir]*cell_size[dir]);
      if( iCell_n0.level_diff() == 1 ) // bigger
          update_minmod(dir, iCell_n0, pos_cell[dir]+1.5*offset[dir]*cell_size[dir]);
      if( iCell_n0.level_diff() == -1 ) // smaller
      {
        foreach_smaller_neighbor<ndim>(iCell_n0, offset, Q, 
          [&](const CellIndex& iCell_neighbor)
        {
          update_minmod(dir, iCell_neighbor, pos_cell[dir]+0.75*offset[dir]*cell_size[dir]);
        });
      }
  };

  process_neighbors(IX, +1);
  process_neighbors(IX, -1);
  process_neighbors(IY, +1);
  process_neighbors(IY, -1);
  if(ndim==3)
  {
    process_neighbors(IZ, +1);
    process_neighbors(IZ, -1);
  }

  PrimState gradP[ndim]{};

  gradP[IX] = grad[IX];
  setPrimitiveState<ndim>(Slope_x, iCell_U, gradP[IX]);

  gradP[IY] = grad[IY];
  setPrimitiveState<ndim>(Slope_y, iCell_U, gradP[IY]);

  if(ndim==3) {
    gradP[IZ] = grad[IZ];
    setPrimitiveState<ndim>(Slope_z, iCell_U, gradP[IZ]);
  }
}

/**
   * Compute Riemann fluxes and update current cell.
   *
   * @param Uin Initial fields
   * @param Uout Fields to update
   * @param Q Primitive fields
   * @param iCell_Q current cell index
   * @param Slope_xyz limited slopes for every cell (in the xyz direction)
   * @param cellmetadata cell meta data to retrieve cell size and position
   * @param dt timestep
   * @param params RiemannParams configuration
   */
template< 
  int ndim,
  typename State >
KOKKOS_INLINE_FUNCTION
void compute_fluxes_and_update( const FieldArray& Uin, const FieldArray& Uout, const GhostedArray& Q, const CellIndex& iCell_Q,
                                const GhostedArray& Slopes_x, const GhostedArray& Slopes_y, const GhostedArray& Slopes_z,
                                const ForeachCell::CellMetaData& cellmetadata, real_t dt, const RiemannParams& params,
                                const BoundaryConditions& bc_manager)
{
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  ForeachCell::CellMetaData::pos_t cell_size = cellmetadata.getCellSize(iCell_Q);
  ForeachCell::CellMetaData::pos_t pos_c = cellmetadata.getCellCenter(iCell_Q);

  PrimState qprim{};
  ConsState qcons{};
  getPrimitiveState<ndim>(Q, iCell_Q, qprim);
  getConservativeState<ndim>(Uin, iCell_Q, qcons);

  /**
   * Solve riemann problem at interface between cells
   * @param qr_c primitive variables for current cell
   * @param qr_n primitive variables for neighbor cell
   * @param dir direction of interface between cells (e.g IX if neighbor is left of cell)
   * @param int sign : -1 neighbor is left of current cell, +1 neighbor is right
   * @return flux between cells
   **/
  auto riemann = [&](PrimState qr_c, PrimState qr_n, ComponentIndex3D dir, int sign)
  {
    PrimState qr_c_swap = swapComponents(qr_c, dir);
    PrimState qr_n_swap = swapComponents(qr_n, dir);

    PrimState &qr_L = (sign<0)?qr_n_swap:qr_c_swap;
    PrimState &qr_R = (sign<0)?qr_c_swap:qr_n_swap;
    ConsState flux = riemann_hydro(qr_L, qr_R, params);

    return swapComponents(flux, dir);
  };

  
  using reconstruct_offset_t = Kokkos::Array<real_t, 3>;

  /**
   * Reconstruct state form cell values and slopes at the iven position
   * 
   * @param q initial cell value
   * @param iCell_U cell position
   * @param Position on cell border where the primitive variables must be reconstructed
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
   * @param dx_over_2 half cell size
   **/
  auto reconstruct_state = [&] ( PrimState q, 
                                 const CellIndex& iCell_U,
                                 reconstruct_offset_t offsets,
                                 ForeachCell::CellMetaData::pos_t cell_size, real_t dt ) -> PrimState
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( ndim==3 || offsets[IZ]==0, "offsets[IZ] should be 0 in 2D" );

    const real_t gamma  = params.gamma0;
    const real_t smallr = params.smallr;
    const real_t smallp = params.smallp;
    
    const real_t dtdx = dt/cell_size[IX];
    const real_t dtdy = dt/cell_size[IY];
    const real_t dtdz = dt/cell_size[IZ];

    PrimState diff_x{};
    PrimState diff_y{};
    PrimState diff_z{};

    getPrimitiveState<ndim>( Slopes_x, iCell_U, diff_x );
    getPrimitiveState<ndim>( Slopes_y, iCell_U, diff_y );
    if (ndim == 3)
      getPrimitiveState<ndim>( Slopes_z, iCell_U, diff_z );

    diff_x *= cell_size[IX];
    diff_y *= cell_size[IY];
    diff_z *= cell_size[IZ];

    auto qs = compute_source<ndim>(q, diff_x, diff_y, diff_z, dtdx, dtdy, dtdz, gamma);
    // reconstruct state on interface
    PrimState qr = qs + diff_x * 0.5 * offsets[IX] + diff_y * 0.5 * offsets[IY] + diff_z * 0.5 * offsets[IZ];
    qr.rho = fmax(smallr, qr.rho);
    qr.p   = fmax(smallp, qr.p);

    return qr;    
  };

  // Process every neighbor in direction dir (sign = Left(-1)/Right(1))
  auto process_dir = [&]( ComponentIndex3D dir, int sign )
  {
    CellIndex::offset_t offset{};
    offset[dir] = sign;
    CellIndex iCell_n0 = iCell_Q.getNeighbor_ghost( offset, Q );

    if( iCell_n0.is_boundary() )
    {
      
      PrimState qr_c = qprim;
      ConsState ur_n = bc_manager.template getBoundaryValue<ndim, State>(Uin, iCell_n0, cellmetadata);
      PrimState qr_n = consToPrim<ndim>(ur_n, params.gamma0);
      ConsState flux = riemann(qr_c, qr_n, dir, sign);

      if (sign == 1 && bc_manager.bc_min[dir] == BC_USER)
        flux = bc_manager.template overrideBoundaryFlux<ndim, State>(flux, qr_c, dir, true);
      else if (sign == -1 && bc_manager.bc_max[dir] == BC_USER)
        flux = bc_manager.template overrideBoundaryFlux<ndim, State>(flux, qr_c, dir, false);
        
      // +- dS / dV 
      real_t scale = -sign * dt / cell_size[dir];     
      qcons += flux*scale;     
    }
    else
    {
       ForeachCell::CellMetaData::pos_t cell_size_n = cellmetadata.getCellSize(iCell_n0);

      if( iCell_n0.level_diff() >= 0 ) // Only one cell
      {
        // 0. retrieve primitive variables in neighbor cell
        PrimState qprim_n{};
        getPrimitiveState<ndim>( Q, iCell_n0, qprim_n );

        // 1. reconstruct primitive variables on both sides of current interface (iface)

        // current cell reconstruction  (primitive variables)
        // Position at which state is reconstructed on current cell border ([-1,1])
        // In this case, neighbor is always bigger or same size : center of face is used
        const reconstruct_offset_t offsets_c{
          (real_t)offset[IX], 
          (real_t)offset[IY], 
          (real_t)offset[IZ] 
        };
        PrimState qr_c = reconstruct_state(qprim, iCell_Q, offsets_c, cell_size, dt);

        // neighbor cell reconstruction (primitive variables)
        // Position at which state is reconstructed on neighbor cell border ([-1,1])
        // In this case, current cell can be smaller or same size
        //  Center of face is used if same size:
        reconstruct_offset_t offsets_n{
          (real_t)-offset[IX], 
          (real_t)-offset[IY], 
          (real_t)-offset[IZ] 
        };
        //  If "current" is smaller than "neighbor", center of "current" cell is used
        if( iCell_n0.level_diff() > 0 )
        {
          ForeachCell::CellMetaData::pos_t pos_n = cellmetadata.getCellCenter(iCell_n0);
          //We need to determine which quadrant of "neighbor"'s face is "current"'s face
          for( int facedir = 0; facedir < ndim; facedir++ )
          if( facedir != dir ) // Foreach direction inside face (orthogonal to dir)
          {
            offsets_n[facedir] += (pos_c[facedir] > pos_n[facedir])? 0.5 : -0.5;
          }
        }
        PrimState qr_n = reconstruct_state(qprim_n, iCell_n0, offsets_n, cell_size_n, dt);

        ConsState flux = riemann(qr_c, qr_n, dir, sign);
        // +- dS / dV 
        real_t scale = -sign * dt / cell_size[dir] ;      
        qcons += flux*scale; 
      }
      else // ( iCell_n0.level_diff() == -1 ) // Multiple smaller neighbors
      {
        // Accumulate fluxes from neighbors of initial cell
        int di_count = (offset[IX]==0)?2:1;
        int dj_count = (offset[IY]==0)?2:1;
        int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
        for( int8_t dk=0; dk<dk_count; dk++ )
        for( int8_t dj=0; dj<dj_count; dj++ )
        for( int8_t di=0; di<di_count; di++ )
        {
            CellIndex iCell_n = iCell_n0.getNeighbor_ghost({di,dj,dk}, Q);
            // 0. retrieve primitive variables in neighbor cell
            PrimState qprim_n{};
            getPrimitiveState<ndim>( Q, iCell_n, qprim_n );

            // 1. reconstruct primitive variables on both sides of current interface (iface)

            // current cell reconstruction  (primitive variables)
            // Position at which state is reconstructed on current cell border ([-1,1])
            // In this case, neighbor is always smaller : center of neighbor face is used
            // Relative position of neighbor can be computed from {di,dj,dk}
            const reconstruct_offset_t offsets_c{
              (offset[IX] != 0) ? (real_t)offset[IX] : di-0.5, 
              (offset[IY] != 0) ? (real_t)offset[IY] : dj-0.5, 
              (ndim==2 || offset[IZ] != 0) ? (real_t)offset[IZ] : dk-0.5
            };
            PrimState qr_c = reconstruct_state(qprim, iCell_Q, offsets_c, cell_size, dt);

            // neighbor cell reconstruction (primitive variables)
            // Position at which state is reconstructed on neighbor cell border ([-1,1])
            // In this case, current cell is bigger than neighbor, center of neighbor is used
            const reconstruct_offset_t offsets_n{
              (real_t)-offset[IX], 
              (real_t)-offset[IY], 
              (real_t)-offset[IZ] 
            };            
            PrimState qr_n = reconstruct_state(qprim_n, iCell_n, offsets_n, cell_size_n, dt);
            ConsState flux = riemann(qr_c, qr_n, dir, sign);
            // +- dS / dV 
            int nneigh = (ndim-1)*2;
            real_t scale = -sign * dt /  (cell_size[dir] * nneigh)  ;      
            qcons += flux*scale; 
        }
      }
    }
  };

  process_dir(IX, +1);
  process_dir(IX, -1);
  process_dir(IY, +1);
  process_dir(IY, -1);
  if(ndim==3)
  {
    process_dir(IZ, +1);
    process_dir(IZ, -1);
  }

  setConservativeState<ndim>(Uout, iCell_Q, qcons);
}

enum VarIndex_gravity {IGX, IGY, IGZ};

/**
 * Applies corrector step for gravity
 * @param Uin Initial values before update
 * @param iCell_Uin Position insides Uin/Uout (non ghosted)
 * @param dt time step
 * @param use_field Get gravity field from Uin
 * @param gx, gy, gz, scalar values when use_field == false
 * @param Uout Updated array after hydro without gravity
 **/
template<int ndim, typename State>
KOKKOS_INLINE_FUNCTION
void apply_gravity_correction( const FieldArray& Uin,
                               const FieldArray& Uin_g,
                               const CellIndex& iCell_Uin,
                               real_t dt,
                               bool use_field,
                               real_t gx, real_t gy, real_t gz,
                               const FieldArray& Uout ){
  
  if(use_field)
  {
    gx = Uin_g.at(iCell_Uin, IGX);
    gy = Uin_g.at(iCell_Uin, IGY);
    if (ndim == 3)
      gz = Uin_g.at(iCell_Uin, IGZ);
  }

  real_t rhoOld = Uin.at(iCell_Uin, State::Irho);
  
  real_t rhoNew = Uout.at(iCell_Uin, State::Irho);
  real_t rhou = Uout.at(iCell_Uin, State::Irho_vx);
  real_t rhov = Uout.at(iCell_Uin, State::Irho_vy);
  real_t ekin_old = rhou*rhou + rhov*rhov;
  real_t rhow;
  
  if (ndim == 3) {
    rhow = Uout.at(iCell_Uin, State::Irho_vz);
    ekin_old += rhow*rhow;
  }
  
  ekin_old = 0.5 * ekin_old / rhoNew;

  rhou += 0.5 * dt * gx * (rhoOld + rhoNew);
  rhov += 0.5 * dt * gy * (rhoOld + rhoNew);

  Uout.at(iCell_Uin, State::Irho_vx) = rhou;
  Uout.at(iCell_Uin, State::Irho_vy) = rhov;
  if (ndim == 3) {
    rhow += 0.5 * dt * gz * (rhoOld + rhoNew);
    Uout.at(iCell_Uin, State::Irho_vz) = rhow;
  }

  // Energy correction should be included in case of self-gravitation ?
  real_t ekin_new = rhou*rhou + rhov*rhov;
  if (ndim == 3)
    ekin_new += rhow*rhow;
  
  ekin_new = 0.5 * ekin_new / rhoNew;
  Uout.at(iCell_Uin, State::Ie_tot) += (ekin_new - ekin_old);
}

}

/**
 * @brief Solves the hydrodynamics equations with a Hancock timestepping on 
 * small blocks.
 * 
 * This solver should only be used for comparison with cell-based AMR
 * as it is the only one allowing for bx=by=bz=1.
*/
template <typename State_>
class HydroUpdate_hancock_oneneighbor : public HydroUpdate{
public: 
  using State = State_;
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  HydroUpdate_hancock_oneneighbor(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : foreach_cell(foreach_cell),
      riemann_params(configMap),
      bc_manager(configMap),
      gx( configMap.getValue<real_t>("gravity", "gx",  0.0) ),
      gy( configMap.getValue<real_t>("gravity", "gy",  0.0) ),
      gz( configMap.getValue<real_t>("gravity", "gz",  0.0) ),
      timers(timers)
  {
    GravityType gravity_type = configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE);
    this->gravity_enabled = gravity_type!=GRAVITY_NONE;
    this->gravity_use_field = gravity_type&GRAVITY_FIELD;
    [[maybe_unused]] bool gravity_use_scalar = gravity_type==GRAVITY_CST_SCALAR;

    DYABLO_ASSERT_HOST_RELEASE( !gravity_enabled || ( gravity_use_field != gravity_use_scalar ),
      "If gravity is on it must either use the force field from U or a constant scalar force field"  );
  }

  void update( UserData& U, ScalarSimulationData& scalar_data) 
  {
    real_t dt = scalar_data.get<real_t>("dt");
    int ndim = foreach_cell.getDim();
    if(ndim==2)
      update_aux<2>(U, dt);
    else if(ndim==3)
      update_aux<3>(U, dt);
    else 
      DYABLO_ASSERT_HOST_RELEASE(false, "invalid ndim = " << ndim);
  }

  template< int ndim>
  void update_aux(  UserData& U,
                    real_t dt) {
    using GhostedArray = ForeachCell::CellArray_global_ghosted;
    const RiemannParams& riemann_params = this->riemann_params;
    const BoundaryConditions& bc_manager = this->bc_manager;
    ForeachCell& foreach_cell = this->foreach_cell;
    int nb_ghosts = 1;
    GhostCommunicator ghost_comm(foreach_cell.get_amr_mesh(), U.getShape(), nb_ghosts );
    bool gravity_use_field = this->gravity_use_field;
    bool gravity_enabled = this->gravity_enabled;
    real_t gx = this->gx;
    real_t gy = this->gy;
    real_t gz = this->gz;

    timers.get("HydroUpdate_hancock_oneneighbor").start();

    auto fields_info = ConsState::getFieldsInfo();
    UserData::FieldAccessor Uin = U.getAccessor( fields_info );
    UserData::FieldAccessor Uin_g;
    if( gravity_use_field )
      Uin_g = U.getAccessor( {{"gx",IGX}, {"gy",IGY}, {"gz",IGZ}} );    
    
    auto fields_info_next = fields_info;
    for( auto& p : fields_info_next )
      p.name += "_next";
    UserData::FieldAccessor Uout = U.getAccessor( fields_info_next );

    FieldManager fm_prim = PrimState::getFieldManager();
    
    GhostedArray Q = foreach_cell.allocate_ghosted_array( "Q", fm_prim );

    // Fill Q with primitive variables
    foreach_cell.foreach_cell("HydroUpdate_hancock_oneneighbor::convertToPrimitives", Q, KOKKOS_LAMBDA(const CellIndex& iCell_Q)
    { 
        computePrimitives<ndim, State>(riemann_params, Uin, iCell_Q, Q);
    });
    // Primitive variables of ghost cells are needed to compute slopes
    ghost_comm.exchange_ghosts(Q);

    // Create arrays to store slopes
    GhostedArray Slopes_x = foreach_cell.allocate_ghosted_array( "Slopes_x", fm_prim );
    GhostedArray Slopes_y = foreach_cell.allocate_ghosted_array( "Slopes_Y", fm_prim );
    GhostedArray Slopes_z;
    if(ndim == 3)
      Slopes_z = foreach_cell.allocate_ghosted_array( "Slopes_z", fm_prim );

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Fill slope arrays
    foreach_cell.foreach_cell("HydroUpdate_hancock_oneneighbor::reconstruct_gradients", Q, KOKKOS_LAMBDA(const CellIndex& iCell_Q)
    { 
        compute_limited_slopes<ndim, State>(Q, iCell_Q, cellmetadata.getCellCenter(iCell_Q), cellmetadata.getCellSize(iCell_Q), Slopes_x, Slopes_y, Slopes_z);
    });
    // Slopes of ghost cells are needed to compute flux
    ghost_comm.exchange_ghosts(Slopes_x);
    ghost_comm.exchange_ghosts(Slopes_y);
    if(ndim == 3)
      ghost_comm.exchange_ghosts(Slopes_z);

    // Compute flux and update Uout
    foreach_cell.foreach_cell("HydroUpdate_hancock_oneneighbor::flux_and_update", Q, KOKKOS_LAMBDA(const CellIndex& iCell_Q)
    { 
        compute_fluxes_and_update<ndim, State>(Uin, Uout, Q, iCell_Q, 
                                               Slopes_x, Slopes_y, Slopes_z,
                                               cellmetadata, dt, riemann_params, bc_manager);
        // Applying correction step for gravity
      if (gravity_enabled)
        apply_gravity_correction<ndim, ConsState>(Uin, Uin_g, iCell_Q, dt, gravity_use_field, gx, gy, gz, Uout);
    });

    clean_negative_primitive_values<ndim, State>(foreach_cell, Uout, riemann_params.gamma0, riemann_params.smallr, riemann_params.smallp);

    timers.get("HydroUpdate_hancock_oneneighbor").stop();
  }
  private:
    ForeachCell& foreach_cell;
    RiemannParams riemann_params;  
    BoundaryConditions bc_manager;
    real_t gx, gy, gz;

    bool gravity_enabled, gravity_use_field;
    
    Timers& timers;
};

} //namespace dyablo 

FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_hancock_oneneighbor<dyablo::HydroState>, 
                  "HydroUpdate_hancock_oneneighbor")
FACTORY_REGISTER( dyablo::HydroUpdateFactory, 
                  dyablo::HydroUpdate_hancock_oneneighbor<dyablo::MHDState>, 
                  "MHDUpdate_hancock_oneneighbor")
