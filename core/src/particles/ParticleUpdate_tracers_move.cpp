#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"

namespace dyablo {

class ParticleUpdate_tracers_move : public ParticleUpdate {
public:
  ParticleUpdate_tracers_move(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers)
  {}

  ~ParticleUpdate_tracers_move() {}

  void update( UserData& U, ScalarSimulationData& scalar_data) 
  {
    timers.get("ParticleUpdate_tracers_move").start();

    const real_t dt = scalar_data.get<real_t>("dt");

    enum VarIndex_rhov{
      ID,IVX,IVY,IVZ
    };

    auto Uin = U.getAccessor( {{"rho", ID}, {"rho_vx", IVX},{"rho_vy", IVY},{"rho_vz", IVZ}} );
    const ForeachParticle::ParticleArray& P = U.getParticleArray( "particles" );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    foreach_particle.foreach_particle( "tracers_update_position", P,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      ForeachCell::CellIndex iCell = cells.getCellFromPos( {P.pos(iPart, IX), P.pos(iPart, IY), P.pos(iPart, IZ)} );

      real_t rho = Uin.at( iCell, ID );
      P.pos(iPart, IX) += dt * Uin.at( iCell, IVX )/rho;
      P.pos(iPart, IY) += dt * Uin.at( iCell, IVY )/rho;
      P.pos(iPart, IZ) += dt * Uin.at( iCell, IVZ )/rho;
    });   

    timers.get("ParticleUpdate_tracers_move").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;  
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory, 
                  dyablo::ParticleUpdate_tracers_move, 
                  "ParticleUpdate_tracers_move")
