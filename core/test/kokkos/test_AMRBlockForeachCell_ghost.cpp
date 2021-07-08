
#include "shared/DyabloSession.hpp"
#include "shared/amr/AMRmesh.h"
#include "shared/amr/LightOctree.h"
#include "shared/utils_hydro.h"
#include "muscl_block/foreach_cell/AMRBlockForeachCell_group.h"
#include "muscl_block/CopyGhostBlockCellData.h"
#include "utils/monitoring/Timers.h"

namespace dyablo { 
namespace muscl_block {
  using AMRBlockForeachCell = AMRBlockForeachCell_group;
  using ForeachCell = AMRBlockForeachCell;
  using GhostedArray = typename AMRBlockForeachCell::CellArray_global_ghosted;
  using GlobalArray = typename AMRBlockForeachCell::CellArray_global;
  using PatchArray = typename AMRBlockForeachCell::CellArray_patch;
  using CellIndex = typename AMRBlockForeachCell::CellIndex;
}// namespace dyablo
}// namespace muscl_block

#include "muscl_block/update/CopyGhostBlockCellData.h"

using namespace dyablo;
using namespace dyablo::muscl_block;

int main(int argc, char *argv[])
{
  shared::DyabloSession mpi_session(argc, argv);
 
  constexpr int ITER = 10;
  constexpr int ndim = 3;
  constexpr int balance_codim = 3;
  constexpr std::array<bool,3> periodic = {true, true, true};
  constexpr uint8_t level_min = 3;
  constexpr uint8_t level_max = 6;
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
  constexpr BoundaryConditionType xbound = BC_PERIODIC;
  constexpr BoundaryConditionType ybound = BC_PERIODIC;
  constexpr BoundaryConditionType zbound = BC_PERIODIC;
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

    // Set all fields used in CopyGhostBlockCellDataFunctor
    HydroParams params;
    params.boundary_type_xmin = xbound;
    params.boundary_type_xmax = xbound;
    params.boundary_type_ymin = ybound;
    params.boundary_type_ymax = ybound;
    params.boundary_type_zmin = zbound;
    params.boundary_type_zmax = zbound;
    params.dimType = THREE_D;
    // Set extra unused fields to avoid -Wmaybe-uninitialized
    params.gx = 0;
    params.gy = 0;
    params.gz = 0;
    params.data_type = 0;
    params.nDim = 0;
    params.level_min = 0;
    params.level_max = 0;
    params.rsst_cfl_enabled = 0;
    params.rsst_enabled = 0;
    params.communicator=0;


    char configmap_cstr[] = 
      "[amr]\n"
      "nbTeams=2048\n";
    int configmap_cstr_len = sizeof(configmap_cstr);

    char* configmap_charptr = configmap_cstr;

    ConfigMap configMap(configmap_charptr, configmap_cstr_len);

    timers.get("run_Kokkos_parallelfor").start();

    for(int i=0; i<ITER; i++)
    {
      InterfaceFlags interface_flags(nbOctsPerGroup);
      for(uint32_t iGroup=0; iGroup<nbOcts/nbOctsPerGroup; iGroup++ )
      {
        dyablo::muscl_block::CopyGhostBlockCellDataFunctor::apply(
                                          lmesh,
                                          configMap,
                                          params,
                                          fm,
                                          {bx,by,bz},
                                          2,
                                          nbOctsPerGroup,
                                          U,
                                          Ughost,
                                          Ugroup, 
                                          iGroup,
                                          interface_flags);
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
        patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
        {
            copyGhostBlockCellData<ndim>(
            Uin, iCell_Ugroup, 
            patch, 
            xmin, ymin, zmin, 
            xmax, ymax, zmax, 
            xbound, ybound, zbound,
            Ugroup);
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
