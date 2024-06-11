#include "Compute_dt_base.h"

#include "utils_hydro.h"

#include "states/State_forward.h"

#include "parabolic/ParabolicTerm_thermal_conduction.h"
#include "parabolic/ParabolicTerm_viscosity.h"

namespace dyablo {

/**
 * @brief Timestep limiter for parabolic terms solved explicitly.
 * 
 * Accounts for thermal conduction and viscosity
 * 
 * The limitation is of the form dt = C min_h(d_h^2/lambda_h)
 * with :
 *  . C a constant factor < 0.5,
 *  . d_h the cell size along direction h
 *  . lambda_h the diffusion coefficient along direction h
 */
class Compute_dt_parabolic : public Compute_dt
{
public:
  Compute_dt_parabolic(   ConfigMap& configMap,
                        ForeachCell& foreach_cell,
                        Timers& timers )
  : foreach_cell(foreach_cell),
    pt_tc(configMap),
    pt_visc(configMap)
  {
    real_t default_cfl = 0.5;
    this->cfl = configMap.getValue<real_t>("dt", "parabolic_cfl", default_cfl);
    
    std::string tc_update_id = configMap.getValue<std::string>("thermal_conduction", "update", "none");
    std::string visc_update_id = configMap.getValue<std::string>("viscosity", "update", "none");

    this->compute_dt_for_tc   = (tc_update_id == "ParabolicUpdate_explicit");
    this->compute_dt_for_visc = (visc_update_id == "ParabolicUpdate_explicit");
  }

  void compute_dt( const UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t dt_local;

    int ndim = foreach_cell.getDim();

    if (ndim == 2)
      dt_local = compute_dt_aux<2>(U);
    else
      dt_local = compute_dt_aux<3>(U);
    
    DYABLO_ASSERT_HOST_RELEASE(dt_local>0, "invalid dt = " << dt_local);

    real_t dt;
    auto communicator = foreach_cell.get_amr_mesh().getMpiComm();
    communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);

    scalar_data.set<real_t>("dt", dt);
  }

  template <int ndim>
  double compute_dt_aux( const UserData& U )
  {
    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    auto pt_tc   = this->pt_tc;
    auto pt_visc = this->pt_visc;
    auto compute_for_tc = this->compute_dt_for_tc;
    auto compute_for_visc = this->compute_dt_for_visc;

    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt_parabolic", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      auto cell_pos  = cells.getCellCenter(iCell);
      real_t dx2 = cell_size[IX]*cell_size[IX];
      real_t dy2 = cell_size[IY]*cell_size[IY];
      real_t dz2 = cell_size[IZ]*cell_size[IZ];
      
      // Computing for thermal conduction
      real_t max_dt_tc = 0.0;
      if (compute_for_tc) {
        real_t kappa = pt_tc.compute_kappa<ndim>(cell_pos);
        max_dt_tc = FMAX(kappa/dx2, kappa/dy2);
        if (ndim == 3)
          max_dt_tc = FMAX(max_dt_tc, kappa/dz2);
      }

      // Computing for viscosity
      real_t max_dt_visc = 0.0;
      if (compute_for_visc) {
        real_t mu = pt_visc.compute_mu<ndim>(cell_pos);
        max_dt_visc = FMAX(mu/dx2, mu/dy2);
        if (ndim == 3)
          max_dt_visc = FMAX(max_dt_visc, mu/dz2);
      }

      inv_dt_update = FMAX( inv_dt_update, max_dt_tc );
      inv_dt_update = FMAX( inv_dt_update, max_dt_visc );
    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt = cfl / inv_dt;
    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);
    return dt;
  }

private:
  ForeachCell& foreach_cell;

  real_t cfl;
  bool compute_dt_for_tc;
  bool compute_dt_for_visc;

  ParabolicTerm_thermal_conduction pt_tc;
  ParabolicTerm_viscosity pt_visc;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_parabolic, "Compute_dt_parabolic" );
