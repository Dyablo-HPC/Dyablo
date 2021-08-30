/**
 * \file test_CopyGhostBlockCellDataHash.cpp
 * \author Pierre Kestener
 * \date April, 26th 2020
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>


#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/FieldManager.h"

#ifdef DYABLO_USE_MPI
#include "utils/mpi/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

#include "muscl/SolverHydroMuscl.h"
#include "muscl_block/SolverHydroMusclBlock.h"

#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyFaceBlockCellDataHash.h"
#include "muscl_block/ConvertToPrimitivesHydroFunctor.h"
#include "muscl_block/utils_block.h"

using Device = Kokkos::DefaultExecutionSpace;

#include <boost/test/unit_test.hpp>

#define IDX(ix,iy,iz) ((ix) + bx_g * (iy) + bx_g*by_g*(iz))


namespace dyablo
{

namespace muscl_block
{

template<int dim>
void
call_CopyFaceBlockCellDataHashFunctor(AMRMetaData<dim> mesh,
                                      ConfigMap configMap, 
                                      HydroParams params,
                                      id2index_t fm,
                                      blockSize_t blockSizes, 
                                      uint32_t ghostWidth,
                                      uint32_t nbOctsPerGroup,
                                      DataArrayBlock U,
                                      DataArrayBlock U_ghost,
                                      DataArrayBlock Ugroup,
                                      uint32_t iGroup,
                                      FlagArrayBlock Interface_flags)
{
};

template<>
void
call_CopyFaceBlockCellDataHashFunctor<2>(AMRMetaData<2> mesh,
                                         ConfigMap configMap, 
                                         HydroParams params,
                                         id2index_t fm,
                                         blockSize_t blockSizes, 
                                         uint32_t ghostWidth,
                                         uint32_t nbOctsPerGroup,
                                         DataArrayBlock U,
                                         DataArrayBlock U_ghost,
                                         DataArrayBlock Ugroup,
                                         uint32_t iGroup,
                                         FlagArrayBlock Interface_flags)
{
  muscl_block::CopyFaceBlockCellDataHashFunctor<2>::apply(mesh,
                                                          configMap,
                                                          params, 
                                                          fm,
                                                          blockSizes,
                                                          ghostWidth,
                                                          nbOctsPerGroup,
                                                          U, 
                                                          U_ghost, 
                                                          Ugroup, 
                                                          iGroup,
                                                          Interface_flags);
}

template<>
void
call_CopyFaceBlockCellDataHashFunctor<3>(AMRMetaData<3> mesh,
                                         ConfigMap configMap, 
                                         HydroParams params,
                                         id2index_t fm,
                                         blockSize_t blockSizes, 
                                         uint32_t ghostWidth,
                                         uint32_t nbOctsPerGroup,
                                         DataArrayBlock U,
                                         DataArrayBlock U_ghost,
                                         DataArrayBlock Ugroup,
                                         uint32_t iGroup,
                                         FlagArrayBlock Interface_flags)
{
  muscl_block::CopyFaceBlockCellDataHashFunctor<3>::apply(mesh,
                                                          configMap,
                                                          params, 
                                                          fm,
                                                          blockSizes,
                                                          ghostWidth,
                                                          nbOctsPerGroup,
                                                          U, 
                                                          U_ghost, 
                                                          Ugroup, 
                                                          iGroup,
                                                          Interface_flags);
}

} // namespace muscl_block

// =======================================================================
// =======================================================================
/**
 * Initialize a DataArrayBlock.
 *
 * All cells inside a given octant are initialized with the octant id.
 * This will ease checking that neighbor information is correctly
 * computed.
 *
 * \param[out] data array to initialize
 * \param[in]  nbOcts is the total number of octants
 * \param[in]  params holds hydro parameters
 * \param[in]  blockSizes number of cells per octant (or block)
 * \param[in]  ghostwidth width of the ghost cells around a block
 * \param[in]  fm field manager
 */
void init_U(DataArrayBlock U,
            uint32_t nbOcts,
            HydroParams params,
            muscl_block::blockSize_t blockSizes,
            uint32_t ghostWidth,
            id2index_t fm)
{

  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  uint32_t nbTeams_ = 4;

  team_policy_t policy (nbTeams_,
                        Kokkos::AUTO() /* team size chosen by kokkos */);


  uint32_t bx = blockSizes[IX];
  uint32_t by = blockSizes[IY];
  uint32_t bz = blockSizes[IZ];

  uint32_t nbCells = params.dimType == TWO_D ?
    bx*by : bx*by*bz;

  Kokkos::parallel_for(
    "init_U", policy,
    KOKKOS_LAMBDA(const thread_t& member)
    {
      uint32_t iOct = member.league_rank();
      
      while (iOct <  nbOcts) {
        
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, nbCells),
          [&](const int32_t index) {
            U(index, 0, iOct) =
              // something like a cellwise Morton index
              //index + iOct * nbCells;
              iOct;
          }); // end TeamVectorRange
        
        iOct += nbTeams_;
        
      }
      
    });
  
} // init_U

// =======================================================================
// =======================================================================
template<int dim>
void run_test()
{

  /*
   * testing CopyGhostBlockCellDataFunctor
   */
  std::cout << "// =============================================\n";
  std::cout << "// Testing CopyGhostBlockCellDataHashFunctor ...\n";
  std::cout << "// =============================================\n";

  /*
   * create a fake ConfigMap object
   */
  std::string input_file = dim==2 ? "amr/dummy2d.ini" : "amr/dummy3d.ini";
  ConfigMap configMap = broadcast_parameters(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  /*
   * "geometry" setup
   */

  // block ghost width
  uint32_t ghostWidth = configMap.getInteger("amr", "ghostwidth", 2);
  uint32_t& g=ghostWidth; 

  // block sizes
  uint32_t bx = configMap.getInteger("amr", "bx", 0);
  uint32_t by = configMap.getInteger("amr", "by", 0);
  uint32_t bz = configMap.getInteger("amr", "bz", 1);

  // block sizes with ghosts
  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = by + 2 * ghostWidth;
  uint32_t bz_g = bz + 2 * ghostWidth;

  muscl_block::blockSize_t blockSizes, blockSizes_g;
  blockSizes[IX] = bx;
  blockSizes[IY] = by;
  blockSizes[IZ] = bz;

  blockSizes_g[IX] = bx_g;
  blockSizes_g[IY] = by_g;
  blockSizes_g[IZ] = bz_g;

  std::cout << "Using " 
            << "bx=" << bx << " "
            << "by=" << by << " "
            << "bz=" << bz << " "
            << "bx_g=" << bx_g << " "
            << "by_g=" << by_g << " "
            << "bz_g=" << bz_g << " "
            << "ghostwidth=" << ghostWidth << "\n";

  uint32_t nbCellsPerOct = params.dimType == TWO_D ? bx * by : bx * by * bz;
  uint32_t nbCellsPerOct_g =
      params.dimType == TWO_D ? bx_g * by_g : bx_g * by_g * bz_g;

  uint32_t nbOctsPerGroup = dim==2 ? 32 : 64;

  // ==========================================================
  // STAGE 1 : create a PABLO object (only for initialization)
  // ==========================================================

  /*
   * initialize Pablo mesh
   *
   * 
   * 2d mesh should be exactly like this : 16 octants
   *
   *   ___________ __________
   *  |           |     |    |
   *  |           |  14 | 15 |
   *  |    11     |-----+----|
   *  |           |  12 | 13 |
   *  |___________|_____|____|
   *  |     |     | 8|9 |    |
   *  |  2  |  3  | 6|7 | 10 |
   *  |-----+-----|-----+----|
   *  |  0  |  1  |  4  | 5  |
   *  |_____|_____|_____|____|
   *  
   *
   */

  /*
   * 3d mesh should be exactly like this : 43 octants
   *
   *
   *
   *
   *
   *
   *            ________________________
   *           /           /           /
   *          /           /           /|
   *         /   41      /    42     / |
   *        /           /           /  |
   *       /___________/___________/   |
   *      /           / 39  / 40  /|   |
   *     /   32      /____ /_____/ |   |
   *    /           /  37 / 38  /| /   |               ___________________
   *   /___________/_____/_____/ |/|  /|              |    |    |         |
   *   |           |     |     | / | / |              | 38 | 40 |         |
   *   |           | 37  | 38  |/| |/  |              |____|____|    42   |
   *   |    32     |-----|-----| |/|   |       /      |    |    |         |
   *   |           | 33  | 34  | / |   |     <====    | 34 | 36 |         |
   *   |___________|_____|_____|/| |   /       \      |____|____|_________|
   *   |     |     |16|17|     | | |  /               |    |    |    |    |
   *   |  4  |  5  |-----| 20  |  /| /                | 20 | 22 | 29 | 31 |
   *   |_____|_____|12|13|_____| /| /                 |____|____|____|____|
   *   |     |     |     |     |  |/                  |    |    |    |    |
   *   |  0  |  1  |  8  |  9  |  /                   | 9  | 11 | 25 | 27 |
   *   |_____|_____|_____|_____| /                    |____|____|____|____|
   *
   *
   *
   *
   *
   */

  dyablo::AMRmesh amr_mesh(dim);

  // Set 2:1 balance
  // codim 1 ==> balance through faces
  // codim 2 ==> balance through faces and corner
  // codim 3 ==> balance through faces, edges and corner (3D only)
  int codim = 1; 
  amr_mesh.setBalanceCodimension(codim);
  
  uint32_t idx=0;
  amr_mesh.setBalance(idx,true);

  // set periodic border condition
  amr_mesh.setPeriodic(0);
  amr_mesh.setPeriodic(1);
  amr_mesh.setPeriodic(2);
  amr_mesh.setPeriodic(3);
  if (dim==3)
  {
    amr_mesh.setPeriodic(4);
    amr_mesh.setPeriodic(5);
  }

  // step 1
  amr_mesh.adaptGlobalRefine();
  amr_mesh.updateConnectivity();

  // step 2
  amr_mesh.setMarker(1,1);
  amr_mesh.adapt(true);
  amr_mesh.updateConnectivity();

  // step 3
  if (dim==2) 
  {
    amr_mesh.setMarker(3,1);
    amr_mesh.adapt(true);
    amr_mesh.updateConnectivity();
  }
  else
  {
    amr_mesh.setMarker(5,1);
    amr_mesh.adapt(true);
    amr_mesh.updateConnectivity();
  }
  std::cout << "Mesh size is " << amr_mesh.getNumOctants() << "\n";

#if BITPIT_ENABLE_MPI==1
  // (Load)Balance the octree over the processes.
  amr_mesh.loadBalance();
#endif

  // ==================================================
  // STAGE 2 : create a AMRMetaData object 
  //           mirroring the same mesh as PABLO
  // ==================================================

  uint64_t capacity = 1024*1024;
  AMRMetaData<dim> amrMetadata(capacity);

  amrMetadata.report();
  amrMetadata.update_hashmap(amr_mesh);
  amrMetadata.update_neigh_level_status(amr_mesh);
  amrMetadata.report();

  // ==================================================
  // STAGE 3 : fake user data
  // ==================================================

  // just retrieve a field manager
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);
  auto fm = fieldMgr.get_id2index();

  /*
   * allocate/initialize fake data U
   */
  uint32_t nbOcts = amr_mesh.getNumOctants();
  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  DataArrayBlock U = DataArrayBlock("U", nbCellsPerOct, params.nbvar, nbOcts);

  init_U(U, nbOcts, params, blockSizes, ghostWidth, fm);

  uint32_t nghosts = amr_mesh.getNumGhosts();
  DataArrayBlock Ughost = DataArrayBlock("Ughost", nbCellsPerOct, params.nbvar, nghosts);

  /*
   * allocate/initialize Ugroup
   *
   * We assume total number of octant is sufficiently small to assume
   * a single group is enough.
   */

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";

  DataArrayBlock Ugroup = DataArrayBlock("Ugroup", nbCellsPerOct_g, params.nbvar, nbOctsPerGroup);
  uint32_t iGroup = 0;

  //uint8_t nfaces = (params.dimType == TWO_D ? 4 : 6);
  FlagArrayBlock Interface_flags = FlagArrayBlock("Interface Flags", nbOctsPerGroup);
  
  // first copy inner cells
  std::cout << "=========================================\n";
  std::cout << "Testing CopyInnerBlockCellDataFunctor....\n";
  std::cout << "=========================================\n";
  muscl_block::CopyInnerBlockCellDataFunctor::apply(configMap, 
                                                    params,
                                                    fm,
                                                    blockSizes,
                                                    ghostWidth, 
                                                    nbOcts,
                                                    nbOctsPerGroup,
                                                    U, 
                                                    Ugroup,
                                                    iGroup);

  std::cout << "=============================================\n";
  std::cout << "Testing CopyFaceBlockCellDataHashFunctor.....\n";
  std::cout << "=============================================\n";
  {

    if (dim==2) {
      // print mesh
      std::cout << "  2D mesh connectivity        \n";
      std::cout << "   ___________ __________     \n";
      std::cout << "  |           |     |    |    \n";
      std::cout << "  |           |  14 | 15 |    \n";
      std::cout << "  |    11     |-----+----|    \n";
      std::cout << "  |           |  12 | 13 |    \n";
      std::cout << "  |___________|_____|____|    \n";
      std::cout << "  |     |     | 8|9 |    |    \n";
      std::cout << "  |  2  |  3  | 6|7 | 10 |    \n";
      std::cout << "  |-----+-----|-----+----|    \n";
      std::cout << "  |  0  |  1  |  4  | 5  |    \n";
      std::cout << "  |_____|_____|_____|____|    \n";
      std::cout << "                              \n";
    }
    else 
    {
      std::cout << "            ________________________                                        \n";
      std::cout << "           /           /           /|                                       \n";
      std::cout << "          /           /           / |                                       \n";
      std::cout << "         /   41      /    42     /  |                                       \n";
      std::cout << "        /           /           /   |                                       \n";
      std::cout << "       /___________/___________/    |                                       \n";
      std::cout << "      /           / 39  / 40  / |   |                                       \n";
      std::cout << "     /   32      /____ /_____/  |   |                                       \n";
      std::cout << "    /           /  37 / 38  / | |   |               ___________________     \n";
      std::cout << "   /___________/_____/_____/  |/|  /|              |    |    |         |    \n";
      std::cout << "   |           |     |     | /| | / |              | 38 | 40 |         |    \n";
      std::cout << "   |           | 37  | 38  |/ | |/| |              |____|____|    42   |    \n";
      std::cout << "   |    32     |-----|-----|  |/| |/|              |    |    |         |    \n";
      std::cout << "   |           | 33  | 34  |  / | | |      <====   | 34 | 36 |         |    \n";
      std::cout << "   |___________|_____|_____| /| |/ |/              |____|____|_________|    \n";
      std::cout << "   |     |     |16|17|     |  | | /                |    |    |    |    |    \n";
      std::cout << "   |  4  |  5  |-----| 20  |  / |/                 | 20 | 22 | 29 | 31 |    \n";
      std::cout << "   |_____|_____|12|13|_____| /| /                  |____|____|____|____|    \n";
      std::cout << "   |     |     |     |     |  |/                   |    |    |    |    |    \n";
      std::cout << "   |  0  |  1  |  8  |  9  |  /                    | 9  | 11 | 25 | 27 |    \n";
      std::cout << "   |_____|_____|_____|_____| /                     |____|____|____|____|    \n";
      std::cout << "                                                                            \n";
    }

    // expected results
    // first dimension is octant id : 0 to nbOcts-1
    // second dimension enumerate neighbors in Morton order 
    //    the maximun number of neighbors is nbfaces * 2^(dim-1)
    // value is the neighbor octant id 
    Kokkos::View<real_t**, Kokkos::HostSpace> mesh_connectivity("mesh_connectivity", nbOcts, dim==2 ? 4 * 2 : 6 * 4);
    if (dim == 2) {

      mesh_connectivity(0,0) = 5;
      mesh_connectivity(0,1) = 5;
      mesh_connectivity(0,2) = 1;
      mesh_connectivity(0,3) = 1;
      mesh_connectivity(0,4) = 11;
      mesh_connectivity(0,5) = 11;
      mesh_connectivity(0,6) = 2;
      mesh_connectivity(0,7) = 2;

      mesh_connectivity(1,0) = 0;
      mesh_connectivity(1,1) = 0;
      mesh_connectivity(1,2) = 4;
      mesh_connectivity(1,3) = 4;
      mesh_connectivity(1,4) = 11;
      mesh_connectivity(1,5) = 11;
      mesh_connectivity(1,6) = 3;
      mesh_connectivity(1,7) = 3;

      mesh_connectivity(2,0) = 10;
      mesh_connectivity(2,1) = 10;
      mesh_connectivity(2,2) = 3;
      mesh_connectivity(2,3) = 3;
      mesh_connectivity(2,4) = 0;
      mesh_connectivity(2,5) = 0;
      mesh_connectivity(2,6) = 11;
      mesh_connectivity(2,7) = 11;

      mesh_connectivity(3,0) = 2;
      mesh_connectivity(3,1) = 2;
      mesh_connectivity(3,2) = 6;
      mesh_connectivity(3,3) = 8;
      mesh_connectivity(3,4) = 1;
      mesh_connectivity(3,5) = 1;
      mesh_connectivity(3,6) = 11;
      mesh_connectivity(3,7) = 11;

      mesh_connectivity(4,0) = 1;
      mesh_connectivity(4,1) = 1;
      mesh_connectivity(4,2) = 5;
      mesh_connectivity(4,3) = 5;
      mesh_connectivity(4,4) = 14;
      mesh_connectivity(4,5) = 14;
      mesh_connectivity(4,6) = 6;
      mesh_connectivity(4,7) = 7;

      mesh_connectivity(5,0) = 4;
      mesh_connectivity(5,1) = 4;
      mesh_connectivity(5,2) = 0;
      mesh_connectivity(5,3) = 0;
      mesh_connectivity(5,4) = 15;
      mesh_connectivity(5,5) = 15;
      mesh_connectivity(5,6) = 10;
      mesh_connectivity(5,7) = 10;

      mesh_connectivity(6,0) = 3;
      mesh_connectivity(6,1) = 3;
      mesh_connectivity(6,2) = 7;
      mesh_connectivity(6,3) = 7;
      mesh_connectivity(6,4) = 4;
      mesh_connectivity(6,5) = 4;
      mesh_connectivity(6,6) = 8;
      mesh_connectivity(6,7) = 8;

      mesh_connectivity(7,0) = 6;
      mesh_connectivity(7,1) = 6;
      mesh_connectivity(7,2) = 10;
      mesh_connectivity(7,3) = 10;
      mesh_connectivity(7,4) = 4;
      mesh_connectivity(7,5) = 4;
      mesh_connectivity(7,6) = 9;
      mesh_connectivity(7,7) = 9;

      mesh_connectivity(8,0) = 3;
      mesh_connectivity(8,1) = 3;
      mesh_connectivity(8,2) = 9;
      mesh_connectivity(8,3) = 9;
      mesh_connectivity(8,4) = 6;
      mesh_connectivity(8,5) = 6;
      mesh_connectivity(8,6) = 12;
      mesh_connectivity(8,7) = 12;

      mesh_connectivity(9,0) = 8;
      mesh_connectivity(9,1) = 8;
      mesh_connectivity(9,2) = 10;
      mesh_connectivity(9,3) = 10;
      mesh_connectivity(9,4) = 7;
      mesh_connectivity(9,5) = 7;
      mesh_connectivity(9,6) = 12;
      mesh_connectivity(9,7) = 12;

      mesh_connectivity(10,0) = 7;
      mesh_connectivity(10,1) = 9;
      mesh_connectivity(10,2) = 2;
      mesh_connectivity(10,3) = 2;
      mesh_connectivity(10,4) = 5;
      mesh_connectivity(10,5) = 5;
      mesh_connectivity(10,6) = 13;
      mesh_connectivity(10,7) = 13;

      mesh_connectivity(11,0) = 13;
      mesh_connectivity(11,1) = 15;
      mesh_connectivity(11,2) = 12;
      mesh_connectivity(11,3) = 14;
      mesh_connectivity(11,4) = 2;
      mesh_connectivity(11,5) = 3;
      mesh_connectivity(11,6) = 0;
      mesh_connectivity(11,7) = 1;

      mesh_connectivity(12,0) = 11;
      mesh_connectivity(12,1) = 11;
      mesh_connectivity(12,2) = 13;
      mesh_connectivity(12,3) = 13;
      mesh_connectivity(12,4) = 8;
      mesh_connectivity(12,5) = 9;
      mesh_connectivity(12,6) = 14;
      mesh_connectivity(12,7) = 14;

      mesh_connectivity(13,0) = 12;
      mesh_connectivity(13,1) = 12;
      mesh_connectivity(13,2) = 11;
      mesh_connectivity(13,3) = 11;
      mesh_connectivity(13,4) = 10;
      mesh_connectivity(13,5) = 10;
      mesh_connectivity(13,6) = 15;
      mesh_connectivity(13,7) = 15;

      mesh_connectivity(14,0) = 11;
      mesh_connectivity(14,1) = 11;
      mesh_connectivity(14,2) = 15;
      mesh_connectivity(14,3) = 15;
      mesh_connectivity(14,4) = 12;
      mesh_connectivity(14,5) = 12;
      mesh_connectivity(14,6) = 4;
      mesh_connectivity(14,7) = 4;

      mesh_connectivity(15,0) = 14;
      mesh_connectivity(15,1) = 14;
      mesh_connectivity(15,2) = 11;
      mesh_connectivity(15,3) = 11;
      mesh_connectivity(15,4) = 13;
      mesh_connectivity(15,5) = 13;
      mesh_connectivity(15,6) = 5;
      mesh_connectivity(15,7) = 5;

    } else { // dim == 3
    
      // TODO
      // TODO : add results for 3D
      // TODO

    }

    muscl_block::call_CopyFaceBlockCellDataHashFunctor(amrMetadata,
                                                       configMap,
                                                       params, 
                                                       fm,
                                                       blockSizes,
                                                       ghostWidth,
                                                       nbOctsPerGroup,
                                                       U, 
                                                       Ughost, 
                                                       Ugroup, 
                                                       iGroup,
                                                       Interface_flags);
    

    std::cout << "Copy Device data to host....\n";
    auto Ugroup_host = Kokkos::create_mirror(Ugroup);


    Kokkos::deep_copy(Ugroup_host, Ugroup);

    std::cout << "Start checking results (cell to neighbor connectivity)....\n";
    for (uint32_t iOct = 0; iOct<nbOcts; ++iOct)
    {
      std::cout << "iOct=" << iOct << " // ==================================\n";

      if (bz>1) 
      {
      
        //for (uint32_t iz = 0; iz < bz_g; ++iz)
        uint32_t iz = bz_g/2;
        {
          for (uint32_t iy2 = 0; iy2 < by_g; ++iy2)
          {
            uint32_t iy = by_g-1 - iy2;
            for (uint32_t ix = 0; ix < bx_g; ++ix)
            {
              uint32_t index = ix + bx_g * (iy + by_g * iz);
              std::cout << std::right << std::setw(6) << Ugroup_host(index, 0, iOct) << " ";
            }
            std::cout << "\n";
          }
          std::cout << "\n";
        }
        
      } 
      else 
      {
        //
        // WARNING reverse order on Y-axis, so that it's simpler
        // to make a visual comparison with the mesh
        //
        for (uint32_t iy2 = 0; iy2 < by_g; ++iy2)
        {
          uint32_t iy = by_g-1 - iy2;
          
          for (uint32_t ix = 0; ix < bx_g; ++ix)
          {
            uint32_t index = ix + bx_g * iy;
            std::cout << std::right << std::setw(6) << Ugroup_host(index, 0, iOct) << " ";
          }
          std::cout << "\n";
        }

        BOOST_CHECK_CLOSE(Ugroup_host(0+bx_g*2,0,iOct), mesh_connectivity(iOct,0), 0.1);
        BOOST_CHECK_CLOSE(Ugroup_host(0+bx_g*4,0,iOct), mesh_connectivity(iOct,1), 0.1);
        BOOST_CHECK_CLOSE(Ugroup_host(0+bx+g+bx_g*2,0,iOct), mesh_connectivity(iOct,2), 0.1);
        BOOST_CHECK_CLOSE(Ugroup_host(0+bx+g+bx_g*4,0,iOct), mesh_connectivity(iOct,3), 0.1);

        BOOST_CHECK_CLOSE(Ugroup_host(2,0,iOct), mesh_connectivity(iOct,4), 0.1);
        BOOST_CHECK_CLOSE(Ugroup_host(4,0,iOct), mesh_connectivity(iOct,5), 0.1);
        BOOST_CHECK_CLOSE(Ugroup_host(2+(by+g)*bx_g,0,iOct), mesh_connectivity(iOct,6), 0.1);
        BOOST_CHECK_CLOSE(Ugroup_host(4+(by+g)*bx_g,0,iOct), mesh_connectivity(iOct,7), 0.1);

      } // end if bz>1

    } // end for iOct

    if (dim==3)
    {
      
      uint32_t iOct=0;
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2,2)   ,0,iOct), 9, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2),0,iOct), 1, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1,2)   ,0,iOct),23, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2+by,2),0,iOct), 2, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,1)   ,0,iOct),32, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,2+bz),0,iOct), 4, 0.1);

      iOct=1;
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2,2)   ,0,iOct), 0, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2),0,iOct), 8, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1,2)   ,0,iOct),23, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2+by,2),0,iOct), 3, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,1)   ,0,iOct),32, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,2+bz),0,iOct), 5, 0.1);

      iOct=2;
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2,2)   ,0,iOct),11, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2),0,iOct), 3, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1,2)   ,0,iOct), 0, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2+by,2),0,iOct),23, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,1)   ,0,iOct),32, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,2+bz),0,iOct), 6, 0.1);

      iOct=5; // on right face along X, there are 4 neighbors
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2,2)   ,0,iOct), 4, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2),0,iOct),12, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2+by/2,2),0,iOct),14, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2+bz/2),0,iOct),16, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2+by/2,2+bz/2),0,iOct),18, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1,2)   ,0,iOct),23, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2+by,2),0,iOct), 7, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,1)   ,0,iOct), 1, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,2+bz),0,iOct),32, 0.1);

      iOct=21;
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2,2)   ,0,iOct), 7, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2),0,iOct),22, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1,2)   ,0,iOct),14, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2+by,2),0,iOct),28, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,1)   ,0,iOct),10, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,2+bz),0,iOct),35, 0.1);

      iOct=30;
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2,2)   ,0,iOct),23, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2),0,iOct),31, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1,2)   ,0,iOct),28, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2     ,2+by,2     ),0,iOct),12, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx/2,2+by,2     ),0,iOct),13, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2     ,2+by,2+bz/2),0,iOct),16, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx/2,2+by,2+bz/2),0,iOct),17, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,1)   ,0,iOct),26, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,2+bz),0,iOct),42, 0.1);


      iOct=32;
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2     ,     2)   ,0,iOct),34, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2+by/2,2     )   ,0,iOct),36, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2     ,2+bz/2)   ,0,iOct),38, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2+by/2,2+bz/2)   ,0,iOct),40, 0.1);

      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2     ,2     ),0,iOct),33, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2+by/2,2     ),0,iOct),35, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2     ,2+bz/2),0,iOct),37, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2+by/2,2+bz/2),0,iOct),39, 0.1);

      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1   ,2),0,iOct),41, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2+by,2),0,iOct),41, 0.1);

      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2     ,2     ,1)   ,0,iOct), 4, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx/2,2     ,1)   ,0,iOct), 5, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2     ,2+by/2,1)   ,0,iOct), 6, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx/2,2+by/2,1)   ,0,iOct), 7, 0.1);

      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2     ,2     ,2+bz),0,iOct), 0, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx/2,2     ,2+bz),0,iOct), 1, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2     ,2+by/2,2+bz),0,iOct), 2, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx/2,2+by/2,2+bz),0,iOct), 3, 0.1);

      iOct=33;
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2,2)   ,0,iOct),32, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2),0,iOct),34, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1,2)   ,0,iOct),42, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2+by,2),0,iOct),35, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2     ,2     ,1)   ,0,iOct),16, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx/2,2     ,1)   ,0,iOct),17, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2     ,2+by/2,1)   ,0,iOct),18, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx/2,2+by/2,1)   ,0,iOct),19, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,2+bz),0,iOct),37, 0.1);

      iOct=40;
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(1,2,2)   ,0,iOct),39, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2+bx,2,2),0,iOct),32, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,1,2)   ,0,iOct),38, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2+by,2),0,iOct),42, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,1)   ,0,iOct),36, 0.1);
      BOOST_CHECK_CLOSE(Ugroup_host(IDX(2,2,2+bz),0,iOct),11, 0.1);

    }

  } // end testing CopyFaceBlockCellDataFunctor

} // run_test

} // namespace dyablo

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_CASE(test_AMRMetaData2d)
{

  // always run this test
  run_test<2>();
  
} 

BOOST_AUTO_TEST_CASE(test_AMRMetaData3d)
{

  // allow this test to be manually disabled
  // if there is an addition argument, disable
  if (boost::unit_test::framework::master_test_suite().argc==1)
    run_test<3>();
  
} 

BOOST_AUTO_TEST_SUITE_END() /* dyablo */


// old main
#if 0

// =======================================================================
// =======================================================================
// =======================================================================
int main(int argc, char *argv[])
{

  // Create MPI session if MPI enabled
#ifdef DYABLO_USE_MPI
  dyablo::GlobalMpiSession mpiSession(&argc,&argv);
#endif // DYABLO_USE_MPI
  
  Kokkos::initialize(argc, argv);

  int rank = 0;
  int nRanks = 1;

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";

#ifdef DYABLO_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
# ifdef KOKKOS_ENABLE_CUDA
    {

      // To enable kokkos accessing multiple GPUs don't forget to
      // add option "--ndevices=X" where X is the number of GPUs
      // you want to use per node.

      // on a large cluster, the scheduler should assign ressources
      // in a way that each MPI task is mapped to a different GPU
      // let's cross-checked that:
      
      int cudaDeviceId;
      cudaGetDevice(&cudaDeviceId);
      std::cout << "I'm MPI task #" << rank << " (out of " << nRanks << ")"
		<< " pinned to GPU #" << cudaDeviceId << "\n";
      
    }
# endif // KOKKOS_ENABLE_CUDA
#endif // DYABLO_USE_MPI
  }    // end kokkos config

  int testId = 2;
  if (argc>1)
    testId = std::atoi(argv[1]);

  // dim 2
  if (testId == 2)
    dyablo::run_test<2>();
  
  // dim3
  else if (testId == 3)
    dyablo::run_test<3>();

  Kokkos::finalize();

  return EXIT_SUCCESS;
}

#endif // old main
