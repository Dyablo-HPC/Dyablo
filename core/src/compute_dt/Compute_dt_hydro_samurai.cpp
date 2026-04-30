#include "Compute_dt_base.h"

#include "hyperbolic/policy/HyperbolicPolicy_Hydro.h"

namespace dyablo {

/**
 * @brief Timestep limiter for hydrodynamics mimicking what is done in samurai
 * 
 * Limits the timestep according to the CFL condition.
 * The limitation is of the form dt = C * min_h(d_h / |lambda_h|)
 * with :
 *  . C a constant factor < 1,
 *  . d_h the cell_size along direction h,
 *  . lambda_h the maximum signal speed in that direction
 **/
class Compute_dt_hydro_samurai : public Compute_dt
{
private:
using Policy = HyperbolicPolicy_Hydro;

public:
  Compute_dt_hydro_samurai( ConfigMap& configMap,
                    ForeachCell& foreach_cell,
                    Timers& timers )
  : foreach_cell(foreach_cell),
    policy_params(Policy::getParams(configMap))
  {
    real_t default_cfl = 0.5;
    if (configMap.hasValue("hydro", "cfl")) {
      std::cout << "WARNING : hydro/cfl is deprecated in .ini, use dt/hydro_cfl instead !" << std::endl;
      default_cfl = configMap.getValue<real_t>("hydro", "cfl");
    }
    this->cfl = configMap.getValue<real_t>("dt", "hydro_cfl", default_cfl);

    // Verify dt_mhd is enabled if hydro update uses MHD
    // ( "Compute_dt_hydro" used to support MHD and may still be used in outdated .inis )
    bool has_mhd = configMap.getValue<std::string>("hydro", "update", "HydroUpdate_euler").find("MHD") != std::string::npos;
    if (has_mhd && std::is_same_v<Policy, HyperbolicPolicy_Hydro>)
    {
      std::cout << "WARNING : dt/dt_kernel is compute_dt_hydro but MHD policy in use. Use compute_dt_GLMMHD instead !" << std::endl;
      if( ! configMap.getValue<bool>("compute_dt_hydro", "skip_MHD_check", false) )
      {
        DYABLO_ASSERT_HOST_RELEASE( !(has_mhd && std::is_same_v<Policy, HyperbolicPolicy_Hydro>), "dt/dt_kernel is compute_dt_hydro but MHD policy in use. Use compute_dt_GLMMHD instead ! If you think this is an error set compute_dt_hydro/skip_MHD_check = true" );
      }
    }
  }

  void compute_dt( UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t dt_local;
    dt_local = compute_dt_aux(U, scalar_data);
    
    DYABLO_ASSERT_HOST_RELEASE(dt_local>0, "invalid dt = " << dt_local);

    real_t dt;
    auto communicator = foreach_cell.get_amr_mesh().getMpiComm();
    communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);

    scalar_data.set<real_t>("dt", dt);
  }

  double compute_dt_aux( UserData& U, ScalarSimulationData& scalar_data )
  {
    using PrimState = typename Policy::PrimState;
    using ConsState = typename Policy::ConsState;

    Policy policy{ policy_params, scalar_data };

    int ndim = foreach_cell.getDim();
    real_t gamma0 = policy_params.policy_params.gamma0;

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    UserData::FieldAccessor Uin = policy.getUin(U);

    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];
      
      ConsState uLoc = policy.getConsState(Uin, iCell);
      PrimState qLoc = policy.consToPrim(uLoc);

      const real_t cs = sqrt(qLoc.p * gamma0 / qLoc.rho);

      real_t vx = cs + FABS(qLoc.u);
      real_t vy = cs + FABS(qLoc.v);
      real_t vz = (ndim==2)? 0 : cs + FABS(qLoc.w);

      inv_dt_update = Kokkos::max(inv_dt_update, Kokkos::max(vx/dx, Kokkos::max(vy/dy, vz/dz)));

    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt = cfl / inv_dt;
    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);
    return dt;
  }

private:
  ForeachCell& foreach_cell;
  typename Policy::Params policy_params;

  real_t cfl;
};

} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, 
                  dyablo::Compute_dt_hydro_samurai, 
                  "Compute_dt_hydro_samurai" );
