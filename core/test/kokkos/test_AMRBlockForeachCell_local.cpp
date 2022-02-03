
#include "DyabloSession.hpp"
#include "amr/AMRmesh.h"
#include "amr/LightOctree.h"
#include "foreach_cell/AMRBlockForeachCell_group.h"
namespace dyablo { 

  using AMRBlockForeachCell = AMRBlockForeachCell_group;
}
}
#include "utils/monitoring/Timers.h"


using namespace dyablo;
using namespace dyablo::muscl_block;

int main(int argc, char *argv[])
{
  DyabloSession mpi_session(argc, argv);
 
  constexpr int ITER = 100;
  constexpr int ndim = 3;
  constexpr int balance_codim = 3;
  constexpr std::array<bool,3> periodic = {true, true, true};
  constexpr uint8_t level_min = 4;
  constexpr uint8_t level_max = 7;
  constexpr uint32_t bx = 8;
  constexpr uint32_t by = 8;
  constexpr uint32_t bz = 8;
  constexpr uint32_t nbCellPerBlock = bx*by*bz;
  constexpr real_t xmin = 0.0;
  constexpr real_t ymin = 0.0;
  constexpr real_t zmin = 0.0;
  constexpr real_t xmax = 1.0;
  constexpr real_t ymax = 1.0;
  constexpr real_t zmax = 1.0;
  constexpr uint32_t nbOctsPerGroup = 2048;

  Timers timers;

  timers.get("init").start();

  AMRmesh pmesh( ndim, balance_codim, periodic, level_min, level_max );
  {
    for(uint8_t i=0; i<level_min; i++)
      pmesh.adaptGlobalRefine();
    
    for(uint8_t i=level_min; i<level_max; i++)
    {
      for(uint32_t iOct = 2*pmesh.getNumOctants()/5 ; iOct<3*pmesh.getNumOctants()/5; iOct++)
        pmesh.setMarker(iOct, 1);
      pmesh.adapt();
    }
  }
  const LightOctree& lmesh = pmesh.getLightOctree();

  int nbOcts = pmesh.getNumOctants();
  int nbGhosts = pmesh.getNumGhosts();

  id2index_t fm = {};
  constexpr int nbFields = 4;
  fm[ID]=0; fm[IU]=1; fm[IV]=2; fm[IW]=3;
  DataArrayBlock U("U", nbCellPerBlock, nbFields, nbOcts);
  DataArrayBlock Ughost("Ughost", nbCellPerBlock, nbFields, nbGhosts);
  DataArrayBlock U2("U2", nbCellPerBlock, nbFields, nbOcts);

  timers.get("init").stop();

  {
    DataArrayBlock Ugroup("U", (bx+2)*(by+2)*(bz+2), nbFields, nbOcts);

    timers.get("run_Kokkos_parallelfor").start();

    for(int i=0; i<ITER; i++)
    {
      for(uint32_t iGroup=0; iGroup<nbOcts/nbOctsPerGroup; iGroup++ )
      {
        uint32_t nbOctsInGroup = std::min(nbOctsPerGroup, nbOcts-iGroup*nbOctsPerGroup);

        Kokkos::parallel_for( "Kokkos_parallelfor_test",
          nbOctsInGroup * nbCellPerBlock,
          KOKKOS_LAMBDA( uint32_t index )
        {
          uint32_t iOct = iGroup*nbOctsPerGroup + index/nbCellPerBlock;
          uint32_t iCell = index%nbCellPerBlock;

          uint32_t i = iCell%bx;
          uint32_t j = (iCell%(bx*by))/bx;
          uint32_t k = iCell/(bx*by);
          uint32_t iCell_Ugroup = (i+1) + (j+1)*(bx+2) + (k+1)*(bx+2)*(by+2);
          real_t u_minus = Ugroup(iCell_Ugroup - 1            , fm[IU], iOct );
          real_t u_plus  = Ugroup(iCell_Ugroup + 1            , fm[IU], iOct );
          real_t v_minus = Ugroup(iCell_Ugroup - (bx+2)       , fm[IV], iOct );
          real_t v_plus  = Ugroup(iCell_Ugroup + (bx+2)       , fm[IV], iOct );
          real_t w_minus = Ugroup(iCell_Ugroup - (bx+2)*(by+2), fm[IW], iOct );
          real_t w_plus  = Ugroup(iCell_Ugroup + (bx+2)*(by+2), fm[IW], iOct );

          U2( iCell, fm[ID], iOct ) = u_minus*u_minus + u_plus*u_plus + 
                                    v_minus*v_minus + v_plus*v_plus + 
                                    w_minus*w_minus + w_plus*w_plus;
        });
      }
    }

    timers.get("run_Kokkos_parallelfor").stop();

    auto U2_host = Kokkos::create_mirror_view(U2);
    Kokkos::deep_copy(U2_host, U2);
    std::cout << U2_host(0,0,0) << std::endl;
  }

  { // Run with ForeachCell
    using ForeachCell = AMRBlockForeachCell;
    using PatchArray = AMRBlockForeachCell::CellArray_patch;
    using GlobalArray = AMRBlockForeachCell::CellArray_global;
    using GhostArray = AMRBlockForeachCell::CellArray_global_ghosted;
    using CellIndex = AMRBlockForeachCell::CellIndex;
    using o_t = CellIndex::offset_t;

    ForeachCell foreach_cell({
      lmesh.getNdim(),
      lmesh, 
      bx, by, bz, 
      (xmax - xmin)/bx, (ymax - ymin)/by, (zmax - zmin)/bz,
      nbOctsPerGroup
    });


    GhostArray Uin =  foreach_cell.get_ghosted_array(U, Ughost, lmesh, fm);
    GlobalArray Uout = foreach_cell.get_global_array(U2, 0, 0, 0, fm);
    PatchArray::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 1, 1, 1, fm, nbFields);
    timers.get("run_foreach_cell").start();

    for(int i=0; i<ITER; i++)
    {
      foreach_cell.foreach_patch( "ForeachCell_test",
        PATCH_LAMBDA( const ForeachCell::Patch& patch )
      {
        PatchArray Ugroup = patch.allocate_tmp(Ugroup_);
        
        // Copy non ghosted array Uin into temporary ghosted Ugroup with two ghosts
        patch.foreach_cell(Uin, CELL_LAMBDA(const CellIndex& iCell_Uin)
        {
          CellIndex iCell_Ugroup = Ugroup.convert_index(iCell_Uin);

          real_t u_minus = Ugroup.at(iCell_Ugroup + o_t{-1, 0, 0}, IU );
          real_t u_plus  = Ugroup.at(iCell_Ugroup + o_t{ 1, 0, 0}, IU );
          real_t v_minus = Ugroup.at(iCell_Ugroup + o_t{0, -1, 0}, IV );
          real_t v_plus  = Ugroup.at(iCell_Ugroup + o_t{0,  1, 0}, IV );
          real_t w_minus = Ugroup.at(iCell_Ugroup + o_t{0, 0, -1}, IW );
          real_t w_plus  = Ugroup.at(iCell_Ugroup + o_t{0, 0,  1}, IW );

          Uout.at(iCell_Uin, ID) = u_minus*u_minus + u_plus*u_plus + 
                                    v_minus*v_minus + v_plus*v_plus + 
                                    w_minus*w_minus + w_plus*w_plus;
        });
      });
    }

    auto U2_host = Kokkos::create_mirror_view(U2);
    Kokkos::deep_copy(U2_host, U2);
    std::cout << U2_host(0,0,0) << std::endl;

    timers.get("run_foreach_cell").stop();
  }


  timers.print();
  return 0;
}
