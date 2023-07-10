#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"

namespace dyablo {

class ParticleUpdate_NGP_move : public ParticleUpdate {
public:
  ParticleUpdate_NGP_move(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers),
    data{
      .xmin = configMap.getValue<real_t>("mesh", "xmin", 0.0),
      .xmax = configMap.getValue<real_t>("mesh", "xmax", 1.0),      
      .ymin = configMap.getValue<real_t>("mesh", "ymin", 0.0),
      .ymax = configMap.getValue<real_t>("mesh", "ymax", 1.0),
      .zmin = configMap.getValue<real_t>("mesh", "zmin", 0.0),
      .zmax = configMap.getValue<real_t>("mesh", "zmax", 1.0)
    }
  {}

  ~ParticleUpdate_NGP_move() {}

  void update( UserData& U, ScalarSimulationData& scalar_data) 
  {
    timers.get("ParticleUpdate_NGP_move").start();

    const real_t dt = scalar_data.get<real_t>("dt");

    enum VarIndex_g{
      IGX,IGY,IGZ
    };
    enum VarIndex_particle{
      IVX,IVY,IVZ
    };

    auto Uin = U.getAccessor( {{"gx", IGX},{"gy", IGY},{"gz", IGZ}} );
    const ForeachParticle::ParticleArray& Ppos = U.getParticleArray( "particles" );
    UserData::ParticleAccessor Pdata = U.getParticleAccessor( "particles", {{"vx", IVX},{"vy", IVY},{"vz", IVZ}} );

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    const Data& d = this->data;

    foreach_particle.foreach_particle( "particles_update_position", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      ForeachCell::CellIndex iCell = cells.getCellFromPos( {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)} );

      Pdata.at( iPart, IVX ) += dt * Uin.at( iCell, IGX );
      Pdata.at( iPart, IVY ) += dt * Uin.at( iCell, IGY );
      Pdata.at( iPart, IVZ ) += dt * Uin.at( iCell, IGZ );
      Ppos.pos(iPart, IX) += dt * Pdata.at( iPart, IVX );
      Ppos.pos(iPart, IY) += dt * Pdata.at( iPart, IVY );
      Ppos.pos(iPart, IZ) += dt * Pdata.at( iPart, IVZ );

      // Compute periodic position
      Ppos.pos(iPart, IX) = fmod( (Ppos.pos(iPart, IX) - d.xmin) + (d.xmax-d.xmin) , d.xmax-d.xmin) + d.xmin;
      Ppos.pos(iPart, IY) = fmod( (Ppos.pos(iPart, IY) - d.ymin) + (d.ymax-d.ymin) , d.ymax-d.ymin) + d.ymin;
      Ppos.pos(iPart, IZ) = fmod( (Ppos.pos(iPart, IZ) - d.zmin) + (d.zmax-d.zmin) , d.zmax-d.zmin) + d.zmin;

    });   

    timers.get("ParticleUpdate_NGP_move").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;  
public: //Needed for nvcc
  struct Data {
    real_t xmin,xmax,ymin,ymax,zmin,zmax;
  } data;
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory, 
                  dyablo::ParticleUpdate_NGP_move, 
                  "ParticleUpdate_NGP_move")
