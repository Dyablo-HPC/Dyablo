#include "amr/MapUserData_base.h"

#include "amr/CellIndexRemapper.h"
#include "mpi/ViewCommunicator.h"

namespace dyablo {

/**
 * Implementation of MapUserData using mean of smaller cells when coarseneing
 **/
class MapUserData_mean : public MapUserData{
public: 
  MapUserData_mean(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : foreach_cell(foreach_cell)
  {}
  
  ~MapUserData_mean(){}

  void save_old_mesh() override
  {
    this->lmesh_old = this->foreach_cell.get_amr_mesh().getLightOctree();
  }


  void remap( ForeachCell::CellArray_global_ghosted& Uin, ForeachCell::CellArray_global_ghosted& Uout ) override
  {
    using CellIndex = ForeachCell::CellIndex;
    int ndim = foreach_cell.getDim();
    int nbfields = Uin.nbfields();

    CellIndexRemapper remapper( this->lmesh_old, this->foreach_cell );
    
    auto remap = [&](){
      // Detect if a coarsened octant needs ghost values
      int ghost_coarsen_count = 0;
      
      foreach_cell.reduce_cell( "MapUserData_mean::remap", Uout.getShape(),
        KOKKOS_LAMBDA( const CellIndex& iCell_Uout, int& ghost_coarsen_count )
      {
        CellIndex iCell_Uin = remapper.get_old_cell( iCell_Uout );

        if( iCell_Uin.level_diff() >= 0 )
        {
          for(int ivar=0; ivar<nbfields; ivar++)
            Uout.at_ivar( iCell_Uout, ivar ) = Uin.at_ivar( iCell_Uin, ivar );
        }
        else
        {
          for(int ivar=0; ivar<nbfields; ivar++)
            Uout.at_ivar( iCell_Uout, ivar ) = 0;

          int nsubcells = (ndim-1) * 2 * 2;
          for(int8_t dz=0; dz<(ndim-1); dz++)
            for(int8_t dy=0; dy<2; dy++)
              for(int8_t dx=0; dx<2; dx++)
              {
                CellIndex iCell_Uin_n = iCell_Uin.getNeighbor_ghost({dx,dy,dz}, Uin.getShape());
                ghost_coarsen_count += iCell_Uin_n.iOct.isGhost ? 1 : 0;
                for(int ivar=0; ivar<nbfields; ivar++)
                  Uout.at_ivar( iCell_Uout, ivar ) += Uin.at_ivar( iCell_Uin_n, ivar ) / nsubcells;
              }
        }
      }, ghost_coarsen_count);
      if( ghost_coarsen_count > 0 )
        std::cout << "Warning : detected ghost subcells in coarsened octant during remap - this is so unlikely that it was never tested" << std::endl;

      return ghost_coarsen_count;
    };

    int ghost_coarsen_count_local = remap();
    int ghost_coarsen_count;
    foreach_cell.get_amr_mesh().getMpiComm().MPI_Allreduce( &ghost_coarsen_count_local, &ghost_coarsen_count, 1, MpiComm::MPI_Op_t::SUM);
    if( ghost_coarsen_count > 0 )
    {
      // In rare cases, a coarsened cell could have it's subcells scattered on multiple process
      // If this happens, we perform a complete full-block communication of all ghosts
      // before remapping again
      // TODO : find a test-case that uses that
      // TODO : communicate only needed octants

      ViewCommunicator ghost_comm = ViewCommunicator::from_mesh(foreach_cell.get_amr_mesh());
      ghost_comm.exchange_ghosts<2>( Uin.U, Uin.Ughost );
      remap();
    }
  }

private:
  ForeachCell& foreach_cell;
  LightOctree lmesh_old;
};

} // namespace dyablo;

FACTORY_REGISTER( dyablo::MapUserDataFactory , dyablo::MapUserData_mean, "MapUserData_mean")
