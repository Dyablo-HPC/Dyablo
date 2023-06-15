#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"

namespace dyablo {

class ParticleUpdate_NGP_density : public ParticleUpdate {
public:
  ParticleUpdate_NGP_density(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers)
  {}

  ~ParticleUpdate_NGP_density() {}

  void update( UserData& U, real_t dt) 
  {
    timers.get("ParticleUpdate_NGP_density").start();

    enum VarIndex_g{
      IRho, IRhoG
    };
    enum VarIndex_particle{
      IMass
    };

    auto Uin = U.getAccessor( {{"rho", IRho}, {"rho_g", IRhoG}} );
    const ForeachParticle::ParticleArray& Ppos = U.getParticleArray( "particles" );
    UserData::ParticleAccessor Pdata = U.getParticleAccessor( "particles", {{"mass", IMass}} );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    foreach_cell.foreach_cell( "ParticleUpdate_NGP_density::copy_density", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      Uin.at(iCell, IRhoG) = Uin.at(iCell, IRho);
    });

    foreach_particle.foreach_particle( "ParticleUpdate_NGP_density::projection", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      ForeachCell::CellIndex iCell = cells.getCellFromPos( {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)} );
      auto size = cells.getCellSize( iCell );
      real_t rho_contrib = Pdata.at( iPart, IMass ) / (size[IX]*size[IY]*size[IZ]);

      Kokkos::atomic_add( &Uin.at( iCell, IRhoG ), rho_contrib ) ;
    });   

    timers.get("ParticleUpdate_NGP_density").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;  
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory, 
                  dyablo::ParticleUpdate_NGP_density, 
                  "ParticleUpdate_NGP_density")
