#include "Compute_dt_base.h"

#include "utils_hydro.h"

#include "parabolic/ParabolicTerm_thermal_conduction.h"
#include "parabolic/ParabolicTerm_viscosity.h"

namespace dyablo {

/**
 * @brief Timestep limiter forthe parabolic terms for the Belousov-Zhabotinsky setup
 * 
 * Diffusion operators have coefficients Da, Db, and Dc. Those are used with a traditional
 * parabolic CFL condition to limit the time-step
 */
class Compute_dt_BZ : public Compute_dt
{
public:
  Compute_dt_BZ(   ConfigMap& configMap,
                        ForeachCell& foreach_cell,
                        Timers& timers )
  : foreach_cell(foreach_cell)
  {
    real_t default_cfl = 0.5;
    this->cfl = configMap.getValue<real_t>("dt", "parabolic_cfl", default_cfl);
    this->Da  = configMap.getValue<real_t>("BZ", "Da", 2.5e-3);
    this->Db  = configMap.getValue<real_t>("BZ", "Db", 2.5e-3);
    this->Dc  = configMap.getValue<real_t>("BZ", "Dc", 2.5e-3);
  }

  void compute_dt( UserData& U, ScalarSimulationData& scalar_data )
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

    const real_t max_D = std::max(Da, std::max(Db, Dc));
    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt_parabolic", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx2 = cell_size[IX]*cell_size[IX];
      real_t dy2 = cell_size[IY]*cell_size[IY];
      real_t dz2 = cell_size[IZ]*cell_size[IZ];
      
      real_t max_dt = 0.0;
      max_dt = FMAX(max_D/dx2, max_D/dy2);
      if (ndim == 3)
        max_dt = FMAX(max_dt, max_D/dz2);

      inv_dt_update = FMAX( inv_dt_update, max_dt );
    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt = cfl / inv_dt;
    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);
    return dt;
  }

private:
  ForeachCell& foreach_cell;

  real_t cfl;
  real_t Da, Db, Dc;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_BZ, "Compute_dt_BZ" );
