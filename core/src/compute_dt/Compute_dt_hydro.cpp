#include "Compute_dt_base.h"

#include "utils_hydro.h"

#include "states/State_forward.h"

namespace dyablo {

/**
 * @brief Timestep limiter for (magneto)hydrodynamics
 * 
 * Limits the timestep according to the CFL condition.
 * The limitation is of the form dt = C * min_h(d_h / |lambda_h|)
 * with :
 *  . C a constant factor < 1,
 *  . d_h the cell_size along direction h,
 *  . lambda_h the maximum signal speed in that direction
 **/
class Compute_dt_hydro : public Compute_dt
{
public:
  Compute_dt_hydro( ConfigMap& configMap,
                    ForeachCell& foreach_cell,
                    Timers& timers )
  : foreach_cell(foreach_cell),
    gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallc( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
    has_mhd( configMap.getValue<std::string>("hydro", "update", "HydroUpdate_hancock").find("MHD") != std::string::npos )
  {
    real_t default_cfl = 0.5;
    if (configMap.hasValue("hydro", "cfl")) {
      std::cout << "WARNING : hydro/cfl is deprecated in .ini, use dt/hydro_cfl instead !" << std::endl;
      default_cfl = configMap.getValue<real_t>("hydro", "cfl");
    }
    this->cfl = configMap.getValue<real_t>("dt", "hydro_cfl", default_cfl);
  }

  void compute_dt( const UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t dt_local;
    if( has_mhd )
      dt_local = compute_dt_aux<MHDState>(U);
    else
      dt_local = compute_dt_aux<HydroState>(U);

    DYABLO_ASSERT_HOST_RELEASE(dt_local>0, "invalid dt = " << dt_local);

    real_t dt;
    auto communicator = foreach_cell.get_amr_mesh().getMpiComm();
    communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);

    scalar_data.set<real_t>("dt", dt);
  }

  template< typename State >
  double compute_dt_aux( const UserData& U )
  {
    using PrimState = typename State::PrimState;
    using ConsState = typename State::ConsState;

    int ndim = foreach_cell.getDim();
    real_t gamma0 = this->gamma0;

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    std::vector< UserData::FieldAccessor::FieldInfo> fields_info = ConsState::getFieldsInfo();

    UserData::FieldAccessor Uin = U.getAccessor( ConsState::getFieldsInfo() );

    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];
      
      ConsState uLoc;
      if (ndim == 2)
        getConservativeState<2>(Uin, iCell, uLoc);
      else
        getConservativeState<3>(Uin, iCell, uLoc);
      
      PrimState qLoc = consToPrim<3>(uLoc, gamma0);
      const real_t cs = sqrt(qLoc.p * gamma0 / qLoc.rho);

      real_t vx = cs + FABS(qLoc.u);
      real_t vy = cs + FABS(qLoc.v);
      real_t vz = (ndim==2)? 0 : cs + FABS(qLoc.w);

      inv_dt_update = FMAX( inv_dt_update, vx/dx + vy/dy + vz/dz );

      // TODO : Find a BETTER way to do this !
      // In fine, compute_dt should be templated by the type of State
      // and calculations of inv_dt should be held in a separate State function
      if constexpr (std::is_same_v<State, MHDState>) {
        const real_t Bx = qLoc.Bx;
        const real_t By = qLoc.By;
        const real_t Bz = qLoc.Bz;
        const real_t gr = cs*cs*qLoc.rho;
        const real_t Bt2 [] = {By*By+Bz*Bz,
                               Bx*Bx+Bz*Bz,
                               Bx*Bx+By*By};
        const real_t B2 = Bx*Bx + By*By + Bz*Bz;
        const real_t cf1 = gr-B2;
        const real_t V [] = {qLoc.u, qLoc.v, qLoc.w};
        const real_t D [] = {dx, dy, dz};

        for (int i=0; i < ndim; ++i) {
          const real_t cf2 = gr + B2 + sqrt(cf1*cf1 + 4.0*gr*Bt2[i]);
          const real_t cf = sqrt(0.5 * cf2 / qLoc.rho);

          const real_t cmax = FMAX(FABS(V[i] - cf), FABS(V[i] + cf));
          inv_dt_update = FMAX(inv_dt_update, cmax/D[i]);
        }
     }

    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt = cfl / inv_dt;
    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);
    return dt;
  }

private:
  ForeachCell& foreach_cell;

  real_t cfl;
  real_t gamma0, smallr, smallc;  
  bool has_mhd;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_hydro, "Compute_dt_hydro" );
