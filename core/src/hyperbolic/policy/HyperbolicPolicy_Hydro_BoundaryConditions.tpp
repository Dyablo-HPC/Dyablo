#pragma once

namespace dyablo{

class HyperbolicPolicy_BoundaryConditions_Hydro_Default
{
private:
  using HyperbolicPolicy_State = HyperbolicPolicy_State_Hydro;
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using offset_t      = CellIndex::offset_t;

  Kokkos::Array<BoundaryConditionType, 3> bc_min, bc_max;

public:
  using PrimState = typename HyperbolicPolicy_State::PrimState;
  using ConsState = typename HyperbolicPolicy_State::ConsState;

  static std::string name() {return "default";}

  struct Params{
    Kokkos::Array<BoundaryConditionType, 3> bc_min, bc_max;
  };

  
  static Params getParams( ConfigMap& configMap )
  {
    return {
      .bc_min = {
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)
      },
      .bc_max = {
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING)
      }
    };
  }

  HyperbolicPolicy_BoundaryConditions_Hydro_Default( const Params& params, const ScalarSimulationData& )
  : bc_min(params.bc_min),
    bc_max(params.bc_max)
  {}

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Policy_t      &policy, 
                              const Array_t       &U, 
                              const CellIndex     &iCell_boundary, 
                              const CellMetaData  &metadata) const 
  {
    CellIndex iCell_inside;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_inside, offset);

    auto sign = [](int x){return (x>0)-(x<0);};

    CellIndex::offset_t symmetric_offset {
      (-offset[IX] + sign(offset[IX])), 
      (-offset[IY] + sign(offset[IY])), 
      (-offset[IZ] + sign(offset[IZ]))
    }; 

    CellIndex iCell_sym = iCell_inside + symmetric_offset;
    ConsState u_sym = policy.getConsState( U, iCell_sym );    
    ConsState res = u_sym;

    if ( (offset[IX] > 0 && bc_max[IX] == BC_REFLECTING)
      || (offset[IX] < 0 && bc_min[IX] == BC_REFLECTING) )
    {
        res.rho_u = -u_sym.rho_u;
    }
    if ( (offset[IY] > 0 && bc_max[IY] == BC_REFLECTING)
      || (offset[IY] < 0 && bc_min[IY] == BC_REFLECTING) )
    {
        res.rho_v = -u_sym.rho_v;
    }
    if ( (offset[IZ] > 0 && bc_max[IZ] == BC_REFLECTING)
      || (offset[IZ] < 0 && bc_min[IZ] == BC_REFLECTING) )
    {
        res.rho_w = -u_sym.rho_w;
    }

    return res;
  }

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux(  const Policy_t      &policy, 
                              const Array_t       &U, 
                              const CellIndex     &iCell_boundary, 
                              const PrimState     &q_in_reconstructed,
                              const CellMetaData  &metadata) const 
  {
    CellIndex iCell_ref;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_ref, offset);

    bool dir_IX = offset[IX] == -1 || offset[IX] == 1;
    bool dir_IY = offset[IY] == -1 || offset[IY] == 1;
    bool dir_IZ = offset[IZ] == -1 || offset[IZ] == 1;
    DYABLO_ASSERT_KOKKOS_DEBUG( (int)dir_IX + (int)dir_IY + (int)dir_IZ == 1
                              , "offset is not compatible with getBoundaryFlux" );

    ComponentIndex3D dir = IZ;
    if( dir_IX )
      dir = IX;
    else if( dir_IY )
      dir = IY;
    else if( dir_IZ )
      dir = IZ;
    else
      DYABLO_ASSERT_KOKKOS_DEBUG(false, "Internal error! Should not happen");

    const PrimState& q_in = q_in_reconstructed;

    PrimState q_out = q_in;
    {
      if ( (offset[IX] > 0 && bc_max[IX] == BC_REFLECTING)
        || (offset[IX] < 0 && bc_min[IX] == BC_REFLECTING) )
      {
          q_out.u = -q_in.u;
    }
      if ( (offset[IY] > 0 && bc_max[IY] == BC_REFLECTING)
        || (offset[IY] < 0 && bc_min[IY] == BC_REFLECTING) )
    {
          q_out.v = -q_in.v;
    }
      if ( (offset[IZ] > 0 && bc_max[IZ] == BC_REFLECTING)
        || (offset[IZ] < 0 && bc_min[IZ] == BC_REFLECTING) )
      {
          q_out.w = -q_in.w;
      }
    }    

    const PrimState& qL = (offset[dir] == 1) ? q_in : q_out;
    const PrimState& qR = (offset[dir] == 1) ? q_out : q_in;

    ConsState flux_out = policy.riemann_solver(qL, qR, dir);

    return flux_out;
  }
};

class HyperbolicPolicy_BoundaryConditions_Hydro_DoubleMach
{
private:
  using HyperbolicPolicy_State = HyperbolicPolicy_State_Hydro;
  using CellIndex     = ForeachCell::CellIndex;
  using CellMetaData  = ForeachCell::CellMetaData;
  using offset_t      = CellIndex::offset_t;

  struct Rparams {
    real_t gamma0;
    real_t x0;
    real_t alpha;
    real_t post_rho, post_vel, post_p;
    real_t pre_rho, pre_vel, pre_p;
  } rparams;

  real_t sim_time;

public:
  using PrimState = typename HyperbolicPolicy_State::PrimState;
  using ConsState = typename HyperbolicPolicy_State::ConsState;

  static std::string name() {return "double_mach";}

  using Params = Rparams;
  
  static Params getParams( ConfigMap& configMap )
  {
    return {
      configMap.getValue<real_t>("hydro", "gamma0", 1.4),
      configMap.getValue<real_t>("double_mach", "x0", 0.666666667),
      configMap.getValue<real_t>("double_mach", "alpha", 1.0471975511965976),
      configMap.getValue<real_t>("double_mach", "rho_left", 8.0),
      configMap.getValue<real_t>("double_mach", "vel_left", 8.25),
      configMap.getValue<real_t>("double_mach", "p_left", 116.5),
      configMap.getValue<real_t>("double_mach", "rho_right", 1.4),
      configMap.getValue<real_t>("double_mach", "vel_right", 0.0),
      configMap.getValue<real_t>("double_mach", "p_right", 1.0)
    };
  }

  HyperbolicPolicy_BoundaryConditions_Hydro_DoubleMach( const Params& params, const ScalarSimulationData& simulation_data )
  : rparams( params ),
    sim_time( simulation_data.get<real_t>("time") )
  {}
  
  template < typename Array_t, typename Policy_t >
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryFlux(  const Policy_t      &policy, 
                              const Array_t       &U, 
                              const CellIndex     &iCell_boundary, 
                              const PrimState     &q_in_reconstructed,
                              const CellMetaData  &metadata) const 
{
    ConsState out_val = getBoundaryValue(policy, U, iCell_boundary, metadata);
    CellIndex iCell_inside;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_inside, offset);
    auto in_val = policy.getConsState(U, iCell_inside);

    bool left = (offset[IX] < 0 || offset[IY] < 0); 
    ComponentIndex3D dir = (offset[IX] != 0 ? IX : IY);

    PrimState qL, qR;
    if (left) {
      qL = policy.consToPrim(out_val);
      qR = policy.consToPrim(in_val);
    }
    else {
      qL = policy.consToPrim(in_val);
      qR = policy.consToPrim(out_val);
    }

    return policy.riemann_solver(qL, qR, dir);
  }

  template < typename Array_t, typename Policy_t>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue( const Policy_t      &policy, 
                              const Array_t       &U, 
                              const CellIndex     &iCell_boundary, 
                              const CellMetaData  &metadata) const 
  {
    CellIndex iCell_inside;
    offset_t  offset;    
    iCell_boundary.getBoundaryPosAndOffset(iCell_inside, offset);
    ConsState res;
    auto sign = [](int x){return (x>0)-(x<0);};

    CellIndex::offset_t symmetric_offset {
      (-offset[IX] + sign(offset[IX])), 
      (-offset[IY] + sign(offset[IY])), 
      (-offset[IZ] + sign(offset[IZ]))
    }; 

    // X right boundary -> Absorbing
    if ( offset[IX] > 0 ) {
      CellIndex iCell_sym = iCell_inside + symmetric_offset;
      res = policy.getConsState( U, iCell_sym );    
    }
    // X left boundary -> Post-shock state
    else if ( offset[IX] < 0 ) {
      PrimState q;
      q.rho = rparams.post_rho;
      q.u   = rparams.post_vel*sin(rparams.alpha);
      q.v   = -rparams.post_vel*cos(rparams.alpha);
      q.p   = rparams.post_p;
      res = policy.primToCons(q);
    }
    // Y left boundary -> Postshock state before x0 else Reflection
    else if (offset[IY] < 0) {
      real_t x = metadata.getCellCenter(iCell_inside)[IX];
      if (x < rparams.x0) {
        PrimState q;
        q.rho = rparams.post_rho;
        q.u   = rparams.post_vel*sin(rparams.alpha);
        q.v   = -rparams.post_vel*cos(rparams.alpha);
        q.p   = rparams.post_p;
        res = policy.primToCons(q);
      }
      else {    
        CellIndex iCell_sym = iCell_inside + symmetric_offset;
        ConsState u_sym = policy.getConsState( U, iCell_sym );    
        res = u_sym;
        res.rho_v *= -1.0;
      }
    }
    // Y right boundary -> Pre/post shock state depending on time 
    else if (offset[IY] > 0) {
      real_t xs = 10.0*this->sim_time/sin(rparams.alpha) + rparams.x0 + 1.0/tan(rparams.alpha);
      real_t x = metadata.getCellCenter(iCell_inside)[IX];
      PrimState q;

      // Post-shock
      if (x < xs) {  
        q.rho = rparams.post_rho;
        q.u   = rparams.post_vel*sin(rparams.alpha);
        q.v   = -rparams.post_vel*cos(rparams.alpha);
        q.p   = rparams.post_p;
        res = policy.primToCons(q);
      }
      // Pre-shock
      else {
        q.rho = rparams.pre_rho;
        q.u   = rparams.pre_vel;
        q.v   = rparams.pre_vel;
        q.p   = rparams.pre_p;
        res = policy.primToCons(q);
      }
    }

    return res;
  }
};

} //namespace dyablo