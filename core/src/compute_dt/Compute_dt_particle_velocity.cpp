#include "Compute_dt_base.h"

#include "utils_hydro.h"

#include "states/State_forward.h"

namespace dyablo {


class Compute_dt_particle_velocity : public Compute_dt
{
public:
  Compute_dt_particle_velocity( ConfigMap& configMap,
                                ForeachCell& foreach_cell,
                                Timers& timers )
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    cfl( configMap.getValue<real_t>("dt", "particle_cfl", 0.5) )
  {}

  void compute_dt( const UserData& U, ScalarSimulationData& scalar_data )
  {
    real_t dt_local = dt_local = compute_dt_aux(U);

    DYABLO_ASSERT_HOST_RELEASE(dt_local>0, "invalid dt = " << dt_local);

    real_t dt;
    auto communicator = foreach_cell.get_amr_mesh().getMpiComm();
    communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);

    scalar_data.set<real_t>("dt", dt);
  }

  double compute_dt_aux( const UserData& U )
  {
    int ndim = foreach_cell.getDim();
    
    enum VarIndex_particle{
      IVX,IVY,IVZ
    };
    
    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();
    const ForeachParticle::ParticleArray& Ppos = U.getParticleArray( "particles" );
    UserData::ParticleAccessor Pdata = U.getParticleAccessor( "particles", {{"vx", IVX},{"vy", IVY},{"vz", IVZ}} );

    constexpr real_t small_v = 1.0e-10;
    using pos_t = Kokkos::Array<real_t, 3>;

    real_t inv_dt;
    foreach_particle.reduce_particle( "compute_dt", Pdata.getShape(),
    KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex &iPart, real_t& inv_dt_update )
    {
      pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
      ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );

      pos_t cell_size = cells.getCellSize( iCell );
      
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];

      real_t vx = small_v + Pdata.at( iPart, IVX );
      real_t vy = small_v + Pdata.at( iPart, IVY );
      real_t vz = (ndim == 3 ? small_v + Pdata.at( iPart, IVZ ) : 0.0);
      real_t v  = sqrt(vx*vx+vy*vy+vz*vz);
      real_t dmin = FMIN(dx,dy);
      if (ndim == 3)
        dmin = FMIN(dmin, dz);

      inv_dt_update = FMAX( inv_dt_update, v/dmin);
    }, Kokkos::Max<real_t>(inv_dt));

    real_t dt = cfl / inv_dt;
    DYABLO_ASSERT_HOST_RELEASE(dt>0, "invalid dt = " << dt);
    return dt;
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;

  real_t cfl;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_particle_velocity, "Compute_dt_particle_velocity" );
