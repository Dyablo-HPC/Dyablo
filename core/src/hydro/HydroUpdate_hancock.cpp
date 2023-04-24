#include "HydroUpdate_base.h"
#include "states/State_forward.h"
#include "utils/monitoring/Timers.h"

#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "RiemannSolvers.h"

#include "HydroUpdate_utils.h"

#include "boundary_conditions/BoundaryConditions.h"


namespace dyablo { 

namespace {
  using GhostedArray = ForeachCell::CellArray_global_ghosted;
  using GlobalArray = ForeachCell::CellArray_global;
  using PatchArray = ForeachCell::CellArray_patch;
  using CellIndex = ForeachCell::CellIndex;
  using Uin_t = UserData::FieldAccessor;
  using Uout_t = UserData::FieldAccessor;

  /**
   * @brief Computes slopes and sources along each directions on a cell 
   * 
   * @tparam ndim the number of dimensions 
   * @tparam State the current state being manipulated
   * @param Qgroup Array of primitive variables
   * @param iCell_Sources Index of the current cell in Qgroup
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
  void compute_slopes( const PatchArray& Qgroup, const CellIndex& iCell_Sources, real_t gamma, real_t dx, real_t dy, real_t dz, real_t dt,
                       const PatchArray& Sources, const PatchArray& SlopesX, const PatchArray& SlopesY, const PatchArray& SlopesZ)
  {
    using o_t = typename CellIndex::offset_t;
    using PrimState = typename State::PrimState;

    PrimState sx{}, sy{}, sz{};

    CellIndex ib = Qgroup.convert_index(iCell_Sources);
    PrimState qc{};
    getPrimitiveState<ndim>( Qgroup, ib, qc );

    {
      // neighbor along x axis
      PrimState qm{}, qp{};
      getPrimitiveState<ndim>( Qgroup, ib + o_t{-1, 0, 0}, qm);
      getPrimitiveState<ndim>( Qgroup, ib + o_t{ 1, 0, 0}, qp);     

      sx = compute_slope<ndim>(qm, qc, qp, 1.0, 1.0);
      CellIndex iCell_x = SlopesX.convert_index_ghost(iCell_Sources);
      if(iCell_x.is_valid())
        setPrimitiveState<ndim>(SlopesX, iCell_x, sx);
    }

    {
      // neighbor along y axis
      PrimState qm{}, qp{};
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
      PrimState qm{}, qp{};
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
    PrimState qL_swap = swapComponents(qL, dir);
    PrimState qR_swap = swapComponents(qR, dir);

    // Compute flux (Riemann solver)
    ConsState flux = riemann_hydro(qL_swap, qR_swap, params);
    
    return swapComponents(flux, dir);
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
                      const Uout_t& Uout
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


    PrimState sourceL{}, sourceC{}, sourceR{};
    PrimState slopeL{}, slopeC{}, slopeR{};

    getPrimitiveState<ndim>(Source, iCell_Source_L, sourceL);
    getPrimitiveState<ndim>(Source, iCell_Source_C, sourceC);
    getPrimitiveState<ndim>(Source, iCell_Source_R, sourceR);

    getPrimitiveState<ndim>(Slopes, iCell_Slopes_L, slopeL);
    getPrimitiveState<ndim>(Slopes, iCell_Slopes_C, slopeC);
    getPrimitiveState<ndim>(Slopes, iCell_Slopes_R, slopeR);

    ConsState fluxL = compute_flux<ndim, State>( sourceL, sourceC, slopeL, slopeC, dir, smallr, params );
    ConsState fluxR = compute_flux<ndim, State>( sourceC, sourceR, slopeC, slopeR, dir, smallr, params );

    ConsState umod{};
    getConservativeState<ndim>(Uout, iCell_Uout, umod);
    umod += (fluxL - fluxR) * dtddir;

    setConservativeState<ndim>(Uout, iCell_Uout, umod);
  }

  enum Gravity_VarIndex : VarIndex
  {
    IGX, IGY, IGZ
  };

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
  template<int ndim, typename State_t>
  KOKKOS_INLINE_FUNCTION
  void apply_gravity_correction( const Uin_t& Uin,
                                 const Uin_t& Uin_gravity,
                                 const CellIndex& iCell_Uin,
                                 real_t dt,
                                 bool use_field,
                                 real_t gx, real_t gy, real_t gz,
                                 const Uout_t& Uout ){
    if(use_field)
    {
      gx = Uin_gravity.at(iCell_Uin, Gravity_VarIndex::IGX);
      gy = Uin_gravity.at(iCell_Uin, Gravity_VarIndex::IGY);
      if (ndim == 3)
        gz = Uin_gravity.at(iCell_Uin, Gravity_VarIndex::IGZ);
    }

    real_t rhoOld = Uin.at(iCell_Uin, State_t::VarIndex::Irho);
    
    real_t rhoNew = Uout.at(iCell_Uin, State_t::VarIndex::Irho);
    real_t rhou = Uout.at(iCell_Uin, State_t::VarIndex::Irho_vx);
    real_t rhov = Uout.at(iCell_Uin, State_t::VarIndex::Irho_vy);
    real_t ekin_old = rhou*rhou + rhov*rhov;
    real_t rhow;
    
    if (ndim == 3) {
      rhow = Uout.at(iCell_Uin, State_t::VarIndex::Irho_vz);
      ekin_old += rhow*rhow;
    }
    
    ekin_old = 0.5 * ekin_old / rhoNew;

    rhou += 0.5 * dt * gx * (rhoOld + rhoNew);
    rhov += 0.5 * dt * gy * (rhoOld + rhoNew);

    Uout.at(iCell_Uin, State_t::VarIndex::Irho_vx) = rhou;
    Uout.at(iCell_Uin, State_t::VarIndex::Irho_vy) = rhov;
    if (ndim == 3) {
      rhow += 0.5 * dt * gz * (rhoOld + rhoNew);
      Uout.at(iCell_Uin, State_t::VarIndex::Irho_vz) = rhow;
    }

    // Energy correction should be included in case of self-gravitation ?
    real_t ekin_new = rhou*rhou + rhov*rhov;
    if (ndim == 3)
      ekin_new += rhow*rhow;
    
    ekin_new = 0.5 * ekin_new / rhoNew;
    Uout.at(iCell_Uin, State_t::VarIndex::Ie_tot) += (ekin_new - ekin_old);
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
    timers(timers),
    params(configMap),
    bc_manager(configMap),
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
    UserData& U,
    real_t dt)
  {
    uint32_t ndim = foreach_cell.getDim();
    if(ndim == 2)
      update_aux<2>(U, dt);
    else
      update_aux<3>(U, dt);
  }

  template<int ndim>
  void update_aux(
      UserData& U,
      real_t dt)
  {
    const RiemannParams& params = this->params;  
    const real_t gamma = params.gamma0;
    const double smallr = params.smallr;
    const GravityType gravity_type = this->gravity_type;
    
    auto fields_info = ConsState::getFieldsInfo();

    const Uin_t Uin = U.getAccessor( fields_info );
    auto fields_info_next = fields_info;
    for( auto& p : fields_info_next )
      p.name += "_next";
    Uout_t Uout = U.getAccessor( fields_info_next );

    auto bc_manager = this->bc_manager;

    bool has_gravity = gravity_type!=GRAVITY_NONE;
    bool gravity_use_field = gravity_type&GRAVITY_FIELD;
    [[maybe_unused]] bool gravity_use_scalar = gravity_type==GRAVITY_CST_SCALAR;
    real_t gx = this->gx, gy = this->gy, gz = this->gz;

    DYABLO_ASSERT_HOST_RELEASE( !has_gravity || ( gravity_use_field != gravity_use_scalar ),
      "If gravity is on it must either use the force field from U or a constant scalar force field"  );

    Timers& timers = this->timers; 

    ForeachCell& foreach_cell = this->foreach_cell;

    auto fm_cons = ConsState::getFieldManager().get_id2index();
    auto fm_prim = PrimState::getFieldManager().get_id2index();
    
    // Create abstract temporary ghosted arrays for patches 
    PatchArray::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, fm_cons, State::N);
    PatchArray::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, fm_prim, State::N);
    PatchArray::Ref SlopesX_ = foreach_cell.reserve_patch_tmp("SlopesX", 1, 0, 0, fm_prim, State::N);
    PatchArray::Ref SlopesY_ = foreach_cell.reserve_patch_tmp("SlopesY", 0, 1, 0, fm_prim, State::N);
    PatchArray::Ref SlopesZ_;
    if( ndim == 3 )
      SlopesZ_ = foreach_cell.reserve_patch_tmp("SlopesZ", 0, 0, 1, fm_prim, State::N);
    PatchArray::Ref Sources_ = foreach_cell.reserve_patch_tmp("Sources", 1, 1, (ndim == 3)?1:0, fm_prim, State::N);

    timers.get("HydroUpdate_hancock").start();

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
    const GhostedArray& U_shape = U.getShape();

    Uin_t Uin_gravity;
    if (gravity_use_field) {
      Uin_gravity = U.getAccessor( {
          {"gx", Gravity_VarIndex::IGX}, 
          {"gy", Gravity_VarIndex::IGY},
          {"gz", Gravity_VarIndex::IGZ} 
      });
    }

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
          copyGhostBlockCellData<ndim, State>(
          Uin, iCell_Ugroup, 
          cellmetadata, 
          bc_manager,
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
          Qgroup, iCell_Sources, gamma, size[IX], size[IY], size[IZ], dt,
          Sources, SlopesX, SlopesY, SlopesZ
        );
      });
      
      patch.foreach_cell( U_shape, CELL_LAMBDA(const CellIndex& iCell_Uout)
      {
        auto size = cellmetadata.getCellSize(iCell_Uout);
        ConsState u0{};
        getConservativeState<ndim>(Uin, iCell_Uout, u0);
        setConservativeState<ndim>(Uout, iCell_Uout, u0);
        compute_fluxes<ndim, State>(IX, iCell_Uout, SlopesX, Sources, params, smallr, dt/size[IX], Uout);
        compute_fluxes<ndim, State>(IY, iCell_Uout, SlopesY, Sources, params, smallr, dt/size[IY], Uout);
        if( ndim==3 ) compute_fluxes<ndim, State>(IZ, iCell_Uout, SlopesZ, Sources, params, smallr, dt/size[IZ], Uout);
      
        // Applying correction step for gravity
        if (has_gravity)
         apply_gravity_correction<ndim, ConsState>(Uin, Uin_gravity, iCell_Uout, dt, gravity_use_field, gx, gy, gz, Uout);
      });
    });

    clean_negative_primitive_values<ndim, State>(foreach_cell, Uout, params.gamma0, params.smallr, params.smallp);

    timers.get("HydroUpdate_hancock").stop();

  }

private:
  ForeachCell& foreach_cell;
  
  Timers& timers;  

  RiemannParams params;
  BoundaryConditions bc_manager;

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
