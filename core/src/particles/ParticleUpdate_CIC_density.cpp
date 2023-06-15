#include "ParticleUpdate_base.h"

#include "ForeachParticle.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo {

class ParticleUpdate_CIC_density : public ParticleUpdate {
public:
  using pos_t = Kokkos::Array<real_t, 3>;

  ParticleUpdate_CIC_density(
          ConfigMap& configMap,
          ForeachCell& foreach_cell,
          Timers& timers) 
  : foreach_cell(foreach_cell),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    timers(timers)
  {}

  ~ParticleUpdate_CIC_density() {}

  void update( UserData& U, real_t dt) 
  {
    if( foreach_cell.getDim() == 2 )
      update_aux<2>(U, dt);
    else
      update_aux<3>(U, dt);
  }

  template< int ndim>
  void update_aux( UserData& U, real_t dt) 
  {
    timers.get("ParticleUpdate_CIC_density").start();

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

    foreach_cell.foreach_cell( "ParticleUpdate_CIC_density::copy_density", Uin.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      Uin.at(iCell, IRhoG) = Uin.at(iCell, IRho);
    });
    
    foreach_particle.foreach_particle( "ParticleUpdate_CIC_density::projection", Ppos,
      KOKKOS_LAMBDA( const ForeachParticle::ParticleIndex& iPart )
    {
      real_t part_mass = Pdata.at( iPart, IMass );
      pos_t part_pos = {Ppos.pos(iPart, IX), Ppos.pos(iPart, IY), Ppos.pos(iPart, IZ)};
      ForeachCell::CellIndex iCell = cells.getCellFromPos( part_pos );
      
      pos_t cell_size = cells.getCellSize( iCell );
      pos_t cell_pos = cells.getCellCenter( iCell );
      real_t Vcell = cell_size[IX]*cell_size[IY]*cell_size[IZ];

      pos_t p = { // Local position relative to center [-0.5, 0.5]^3
        (part_pos[IX] - cell_pos[IX])/cell_size[IX], 
        (part_pos[IY] - cell_pos[IY])/cell_size[IY], 
        (part_pos[IZ] - cell_pos[IZ])/cell_size[IZ] 
      };
      auto sign = [](const auto& x) -> int8_t {return (x>=0)?1:-1;};
      ForeachCell::CellIndex::offset_t part_offset = {sign(p[IX]), sign(p[IY]), sign(p[IZ])};
      
      pos_t v_out = {abs(p[IX]), abs(p[IY]), abs(p[IZ])};     // volume fraction in neighbor cells [0,0.5]^3
      pos_t v_in =  {1-v_out[IX], 1-v_out[IY], 1-v_out[IZ]};  // Remaining volume fraction in local cell [0.5,1]^3

      auto apply_rho_contrib = [&]( const ForeachCell::CellIndex::offset_t& offset )
      {
        ForeachCell::CellIndex iCell_neighbor = iCell.getNeighbor_ghost(offset, Uin);
        real_t cx = (offset[IX]==0)?v_in[IX]:v_out[IX];
        real_t cy = (offset[IY]==0)?v_in[IY]:v_out[IY];
        real_t cz = (offset[IZ]==0)?v_in[IZ]:v_out[IZ];
        real_t volume_fraction = cx*cy*cz;

        assert( abs( iCell_neighbor.level_diff() ) <= 1 );

        if( iCell_neighbor.level_diff() >= 0 )
        { // bigger or same size : only one cell to write
          real_t rho_contrib = (part_mass * volume_fraction) / Vcell;
          if( iCell_neighbor.level_diff() == 1 )
          { 
            // Same volume fraction, but Vcell_neighbor is 2^ndim times bigger
            rho_contrib = rho_contrib/( 2*2*(ndim-1) );
          }         
          Kokkos::atomic_add( &Uin.at( iCell_neighbor, IRhoG ), rho_contrib) ;
        }
        else 
        {
          // Smaller : write to all smaller neighbors
          //real_t volume_fraction_smaller = volume_fraction/(2*(ndim-1)); // Mass distributed accross neighbors
          //real_t Vcell_smaller = Vcell/( 2*2*(ndim-1) ) // Volume 2^ndim times smaller
          //real_t rho_contrib_smaller = (part_mass * volume_fraction_smaller) / Vcell_smaller;
          real_t rho_contrib = 2 * (part_mass * volume_fraction) / Vcell;
          foreach_smaller_neighbor<ndim>( iCell_neighbor, offset, Uin.getShape(),
            [&]( const ForeachCell::CellIndex& iCell_sn )
          {
            Kokkos::atomic_add( &Uin.at( iCell_sn, IRhoG ), rho_contrib) ;
          });
        }
      };

      apply_rho_contrib( {0              ,0              ,0              });
      apply_rho_contrib( {part_offset[IX],0              ,0              });
      apply_rho_contrib( {0              ,part_offset[IY],0              });
      apply_rho_contrib( {part_offset[IX],part_offset[IY],0              });
      apply_rho_contrib( {0              ,0              ,part_offset[IZ]});
      apply_rho_contrib( {part_offset[IX],0              ,part_offset[IZ]});
      apply_rho_contrib( {0              ,part_offset[IY],part_offset[IZ]});
      apply_rho_contrib( {part_offset[IX],part_offset[IY],part_offset[IZ]});

    });

    GhostCommunicator ghost_comm(std::shared_ptr<AMRmesh>(&foreach_cell.get_amr_mesh(), [](AMRmesh*){}));
    const auto& Urhog = U.getField( "rho_g" );
    ghost_comm.reduce_ghosts<2>( Urhog.U, Urhog.Ughost);

    timers.get("ParticleUpdate_CIC_density").stop();
  }

private:
  ForeachCell& foreach_cell;
  ForeachParticle foreach_particle;
  Timers& timers;  
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::ParticleUpdateFactory, 
                  dyablo::ParticleUpdate_CIC_density, 
                  "ParticleUpdate_CIC_density")
