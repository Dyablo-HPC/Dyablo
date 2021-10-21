/**
 * \file test_CopyGhostBlockCellData_2D.cpp
 * \author Pierre Kestener, A. Durocher
 * Tests ghost cells copy for block based AMR from `CopyFaceBlockCellDataFunctor`
 * Initializes a mesh with `Implode` initial conditions and calls `CopyFaceBlockCellDataFunctor`
 * Ghost values in 3 octants are verified against the expected results from initial conditions
 * The 3 octants cover different cases (bigger, smaller, same size, boundary) (See declaration of variables iOct1, iOct2, iOct2)
 * \date September, 24th 2019
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>


#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/FieldManager.h"

#include "muscl/SolverHydroMuscl.h"
#include "muscl_block/SolverHydroMusclBlock.h"
#include "muscl_block/init/legacy/InitImplode.h"

#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyGhostBlockCellData.h"
#include "muscl_block/ConvertToPrimitivesHydroFunctor.h"

using Device = Kokkos::DefaultExecutionSpace;

#include <boost/test/unit_test.hpp>

namespace dyablo
{

namespace muscl_block
{

enum NEIGH_SIZE : uint8_t
{
  NEIGH_IS_SMALLER   = 0,
  NEIGH_IS_LARGER    = 1,
  NEIGH_IS_SAME_SIZE = 2
};

// =======================================================================
// =======================================================================
void run_test(int argc, char *argv[])
{
  /*
   * testing CopyGhostBlockCellDataFunctor
   */
  std::cout << "// =========================================\n";
  std::cout << "// Testing CopyGhostBlockCellDataFunctor ...\n";
  std::cout << "// =========================================\n";

  /*
   * read parameter file and initialize a ConfigMap object
   */
  // only MPI rank 0 actually reads input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "./block_data/test_implode_2D_block.ini";
  ConfigMap configMap = broadcast_parameters(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // actually initializing a solver
  // initialize workspace memory (U, U2, ...)
  if (solver_name.find("Muscl_Block") == std::string::npos)
  {

    std::cerr << "Please modify your input parameter file.\n";
    std::cerr << "  solver name must contain string \"Muscl_Block\"\n";

  }

  // just retrieve a field manager
  FieldManager fieldMgr = FieldManager::setup(params, configMap);
  auto fm = fieldMgr.get_id2index();

  /*
   * "geometry" setup
   */

  // block ghost width
  uint32_t ghostWidth = configMap.getInteger("amr", "ghostwidth", 2);

  // block sizes
  uint32_t bx = configMap.getInteger("amr", "bx", 0);
  uint32_t by = configMap.getInteger("amr", "by", 0);
  uint32_t bz = 1;

  // block sizes with ghosts
  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = by + 2 * ghostWidth;
  uint32_t bz_g = 1;

  blockSize_t blockSizes, blockSizes_g;
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

  // create solver members
  //std::unique_ptr<SolverHydroMusclBlock> solver = std::make_unique<SolverHydroMusclBlock>(params, configMap);
  
  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  int ndim = 2;
  params.level_min = 4;
  params.level_max = 5;
  amr_mesh = std::make_shared<AMRmesh>(ndim,ndim,std::array<bool,3>{false,false,false},params.level_min,params.level_max);
  //amr_mesh->setBalanceCodimension(ndim);
  //uint32_t idx = 0;
  //amr_mesh->setBalance(idx,true);
  // amr_mesh->setPeriodic(0);
  // amr_mesh->setPeriodic(1);
  // amr_mesh->setPeriodic(2);
  // amr_mesh->setPeriodic(3);
  //amr_mesh->setPeriodic(4);
  //amr_mesh->setPeriodic(5);
  
  DataArrayBlock Ughost; //solver->Ughost

  amr_mesh->adaptGlobalRefine();
  amr_mesh->adaptGlobalRefine();
  amr_mesh->adaptGlobalRefine();
  amr_mesh->adaptGlobalRefine();
  amr_mesh->setMarker(15,1);
  amr_mesh->setMarker(16,1);
  amr_mesh->setMarker(17,1);
  amr_mesh->setMarker(18,1);
  amr_mesh->setMarker(20,1);
  amr_mesh->setMarker(22,1);
  amr_mesh->setMarker(24,1);
  amr_mesh->setMarker(25,1);
  amr_mesh->setMarker(28,1);

  amr_mesh->adapt();
  amr_mesh->updateConnectivity();

  uint32_t nbOctsPerGroup = 64;
  uint32_t iGroup = 0;
  // Octant that have a "same size" neighbor in all direction
  uint32_t iOct1 = 3;
  // Octant that have at least
  // a "larger size" neighbor in one direction
  uint32_t iOct2 = 15;
  // Octant that have have at least
  // an interface with "smaller size" neighbor in one direction and a periodic boundary
  uint32_t iOct3 = 31;

  std::cout << "Apply initial condition..." << std::endl;

  int nbfields = fieldMgr.nbfields();
  int nbOcts = amr_mesh->getNumOctants();
  uint32_t nbCellsPerOct =
      params.dimType == TWO_D ? bx * by : bx * by * bz;

  DataArrayBlockHost Uhost("Uhost", nbCellsPerOct, nbfields, nbOcts);
  InitImplodeDataFunctor::apply( amr_mesh,
                                params,
                                configMap,
                                fm,
                                blockSizes,
                                Uhost );
  
  DataArrayBlock U("Uhost", nbCellsPerOct, nbfields, nbOcts ); //solver->U
  Kokkos::deep_copy(U, Uhost);

  // by now, init condition must have been called

  uint32_t nbCellsPerOct_g =
      params.dimType == TWO_D ? bx_g * by_g : bx_g * by_g * bz_g;

  /*
   * allocate/initialize Ugroup
   */

  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";
  

  DataArrayBlock Ugroup("Ugroup", nbCellsPerOct_g, params.nbvar, nbOctsPerGroup);
  DataArrayBlockHost UgroupHost("UgroupHost", nbCellsPerOct_g, params.nbvar, nbOctsPerGroup);

  std::cout << "Ugroup sizes = " 
              << Ugroup.extent(0) << " "
              << Ugroup.extent(1) << " "
              << Ugroup.extent(2) << "\n";

  // save solution, just for cross-checking
  //solver->save_solution();

  // Define print functions as lambdas
  // Print info about original local (iGroup set in main()) octant
  auto show_octant = [&](uint32_t iOct_local){  
    //#define DEBUG_PRINT
    #ifdef DEBUG_PRINT
    uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;

    std::cout << "Looking at octant id = " << iOct_global << "\n";
    // octant location
    double x = amr_mesh->getCoordinates(iOct_global)[IX];
    double y = amr_mesh->getCoordinates(iOct_global)[IY];
    std::cout << "Octant location : x=" << x << " y=" << y << "\n";
    auto print_neighbor_status = [&]( int codim, int iface)
    { 
      std::vector<uint32_t> iOct_neighbors;
      std::vector<bool> isghost_neighbors;
      amr_mesh->findNeighbours(iOct_global, iface, codim, iOct_neighbors, isghost_neighbors);
      if (iOct_neighbors.size() == 0)
      {
        std::cout << "( no neigh)";
      }
      else if(iOct_neighbors.size() == 1)
      {
        uint32_t neigh_level = isghost_neighbors[0] ? 
                    amr_mesh->getLevelGhost(iOct_neighbors[0]) : 
                    amr_mesh->getLevel(iOct_neighbors[0]);
        if ( amr_mesh->getLevel(iOct_global) > neigh_level )
          std::cout << "( bigger  )";
        else if ( amr_mesh->getLevel(iOct_global) < neigh_level )
          std::cout << "( smaller )";
        else
          std::cout << "(same size)";
      }
      else
      {
        std::cout << "( smaller )";
      }    
    };
    auto print_neighbor = [&]( int codim, int iface)
    { 
      std::vector<uint32_t> iOct_neighbors;
      std::vector<bool> isghost_neighbors;
      amr_mesh->findNeighbours(iOct_global, iface, codim, iOct_neighbors, isghost_neighbors);
      if (iOct_neighbors.size() == 0)
      {
        std::cout << "(no neigh)";
      }
      else
      {
        std::cout << std::setw(10) << iOct_neighbors[0];
      }
    };
    std::cout << "Octant neighbors :" << std::endl;
    print_neighbor_status(2,0) ; print_neighbor_status(1,2) ; print_neighbor_status(2,1); std::cout << std::endl;
    print_neighbor_status(1,0) ; std::cout << "(         )"; print_neighbor_status(1,1); std::cout << std::endl;
    print_neighbor_status(2,2) ; print_neighbor_status(1,3) ; print_neighbor_status(2,3); std::cout << std::endl;
    std::cout << "Octant neighbors :" << std::endl;
    print_neighbor(2,0) ; print_neighbor(1,2) ; print_neighbor(2,1); std::cout << std::endl;
    print_neighbor(1,0) ; std::cout << "          "; print_neighbor(1,1); std::cout << std::endl;
    print_neighbor(2,2) ; print_neighbor(1,3) ; print_neighbor(2,3); std::cout << std::endl;
    

    // std::cout << "  FACE_LEFT   "; print_neighbor_status(FACE_LEFT);
    // std::cout << "  FACE_RIGHT  "; print_neighbor_status(FACE_RIGHT);
    // std::cout << "  FACE_TOP    "; print_neighbor_status(FACE_TOP);
    // std::cout << "  FACE_BOTTOM "; print_neighbor_status(FACE_BOTTOM);

    std::cout << "Printing Uhost data from iOct = " << iOct_global << "\n";
    for (uint32_t iz=0; iz<bz; ++iz) 
    {
      for (uint32_t iy=0; iy<by; ++iy)
      {
        for (uint32_t ix=0; ix<bx; ++ix)
        {
          uint32_t index = ix + bx*(iy+by*iz);
          printf("%5f ",Uhost(index,fm[ID],iOct_global));
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    #endif
  };

  // Print info about final local octant (with ghosts)
  auto show_octant_group = [&](uint32_t iOct_local){  
    uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;
    // print data from the chosen iGroup 
    std::cout << "Printing Ugroup data from iOct = " << iOct_global << " | iOctLocal = " << iOct_local << " and iGroup = " << iGroup << "\n";       
    for (uint32_t iy = 0; iy < by_g; ++iy)
    {
      for (uint32_t ix = 0; ix < bx_g; ++ix)
      {
        uint32_t index = ix + bx_g * iy;
        printf("%5f ", UgroupHost(index, fm[IP], iOct_local));
      }
      std::cout << "\n";
    } 
  };

  show_octant(iOct1);
  show_octant(iOct2);
  show_octant(iOct3);  

  // first copy inner cells
  CopyInnerBlockCellDataFunctor::apply(configMap, params, fm, 
                                       blockSizes,
                                       ghostWidth, 
                                       nbOcts,
                                       nbOctsPerGroup,
                                       U, Ugroup, iGroup);

  std::cout << "==========================================";
  std::cout << "Testing CopyGhostBlockCellDataFunctor....\n";
  {
    InterfaceFlags interface_flags(nbOctsPerGroup); //solver->interface_flags
    CopyGhostBlockCellDataFunctor::apply(amr_mesh->getLightOctree(),
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
                                        interface_flags);

    Kokkos::deep_copy(UgroupHost, Ugroup);
    
    
    show_octant_group(iOct1);
    show_octant_group(iOct2);
    show_octant_group(iOct3);  

    // CopyFaceBlockCellDataFunctor::apply(solver->amr_mesh,
    //                                     configMap,
    //                                     params, 
    //                                     fm,
    //                                     blockSizes,
    //                                     ghostWidth,
    //                                     nbOctsPerGroup,
    //                                     solver->U, 
    //                                     solver->Ughost, 
    //                                     Ugroup, 
    //                                     iGroup,
    //                                     Interface_flags);
    // CopyCornerBlockCellDataFunctor::apply(solver->amr_mesh,
    //                                     configMap,
    //                                     params, 
    //                                     fm,
    //                                     blockSizes,
    //                                     ghostWidth,
    //                                     nbOctsPerGroup,
    //                                     solver->U, 
    //                                     solver->Ughost, 
    //                                     Ugroup, 
    //                                     iGroup,
    //                                     Interface_flags);
    // show_octant_group(iOct1);
    
    NEIGH_SIZE LARGER = NEIGH_SIZE::NEIGH_IS_LARGER;
    NEIGH_SIZE SMALLER = NEIGH_SIZE::NEIGH_IS_SMALLER;
    // Fetch values from initial conditions and compare to actual value 
    auto check_cell = [&](uint32_t iOct_local, uint32_t ix, uint32_t iy,  NEIGH_SIZE neighbor_size = NEIGH_SIZE::NEIGH_IS_SAME_SIZE )
    { 
      ImplodeParams iParams(configMap);
      uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;
      const real_t octSize = amr_mesh->getSize(iOct_global);
      const real_t cellSize = octSize/bx;
      const real_t x0 = amr_mesh->getCoordinates(iOct_global)[IX];
      const real_t y0 = amr_mesh->getCoordinates(iOct_global)[IY];
      real_t x = x0 + ix*cellSize - ghostWidth*cellSize + cellSize/2;
      real_t y = y0 + iy*cellSize - ghostWidth*cellSize + cellSize/2;
      real_t z = 0;      

      real_t expected;
      if( neighbor_size == LARGER )
      {
        // Works because ghostWidth and blockSizes are pair
        uint32_t ix_larger = (ix/2)*2 + 1;
        uint32_t iy_larger = (iy/2)*2 + 1;
        real_t x_larger = x0 + ix_larger*cellSize - ghostWidth*cellSize;
        real_t y_larger = y0 + iy_larger*cellSize - ghostWidth*cellSize;
        expected = InitImplodeDataFunctor::value( params, iParams, x_larger, y_larger, z )[IP];
        // std::cout << "LARGER neighbor at iOct : " << iOct_global << " pos ghosted : "
        //             "(" << ix_larger << "," << iy_larger << ") = "
        //             "(" << x_larger << "," << y_larger << "," << z << ") :" << std::endl;
      }
      else if( neighbor_size == SMALLER )
      {
        expected = 0;        
        for(int dy=0; dy<2; dy++)
          for(int dx=0; dx<2; dx++)
          {
            real_t x_smaller = x - cellSize/4 + dx*(cellSize/2);
            real_t y_smaller = y - cellSize/4 + dy*(cellSize/2);
            real_t z_smaller = 0;
            expected += InitImplodeDataFunctor::value( params, iParams, x_smaller ,y_smaller ,z_smaller  )[IP];
          }
        expected = expected/4; 
      }
      else
      {
        expected = InitImplodeDataFunctor::value( params, iParams, x,y,z )[IP];
      }
      
      uint32_t index = ix + bx_g * iy;
      real_t& actual = UgroupHost(index, fm[IP], iOct_local);

      BOOST_CHECK_CLOSE(actual, 
                        expected, 0.001);

      //// ---------------------- Debug prints ------------------------
      // if( actual != expected )
      // {
      //   std::cout << "Value at iOct : " << iOct_global << " pos ghosted : "
      //               "(" << ix << "," << iy << ") = "
      //               "(" << x << "," << y << "," << z << ") :" << std::endl;
      //   std::cout << "  Should be (implode) : " << expected << std::endl;
      //   std::cout << "  Is                  : " << actual << std::endl;
      // }
      //// ---------------------- Debug prints ------------------------
    };

    uint32_t iOct = iOct1; // chose an octant which should have a "same size" neighbor in all direction
    uint32_t blocksize = 4;
    for( uint32_t i1=0; i1<blocksize; i1++ )
      for( uint32_t ig=0; ig<ghostWidth; ig++ )
      {
        // Check Bottom border (print top)
        check_cell(iOct,ghostWidth+i1,ig);
        // Check Top border (print bottom)
        check_cell(iOct,ghostWidth+i1,ig+blocksize+ghostWidth);
        // Check Left border
        check_cell(iOct,ig,ghostWidth+i1);
        // Check Right
        check_cell(iOct,ig+blocksize+ghostWidth,ghostWidth+i1);
      }

    for( uint32_t ig1=0; ig1<ghostWidth; ig1++ )
      for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
      {
        //Check Bottom Left
        check_cell(iOct,ig1,ig2);
        //Check Bottom Right
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2);
        //Check Upper Left
        check_cell(iOct,ig1,ig2+blocksize+ghostWidth);
        //Check Upper Right
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth);
      }

    // chose an octant which should have at least
    // a "larger size" neighbor in one direction
    iOct = iOct2;
    for( uint32_t i1=0; i1<blocksize; i1++ )
      for( uint32_t ig=0; ig<ghostWidth; ig++ )
      {
        // Check Bottom border (print top)
        check_cell(iOct,ghostWidth+i1,ig,LARGER);
        // Check Top border (print bottom)
        check_cell(iOct,ghostWidth+i1,ig+blocksize+ghostWidth);
        // Check Left border
        check_cell(iOct,ig,ghostWidth+i1,LARGER);
        // Check Right
        check_cell(iOct,ig+blocksize+ghostWidth,ghostWidth+i1);
      }

    for( uint32_t ig1=0; ig1<ghostWidth; ig1++ )
      for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
      {
        //Check Bottom Left
        check_cell(iOct,ig1,ig2,LARGER);
        //Check Bottom Right
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2,LARGER);
        //Check Upper Left
        check_cell(iOct,ig1,ig2+blocksize+ghostWidth,LARGER);
        //Check Upper Right
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth);
      }

    // chose an octant which should have at least
    // an interface with "smaller size" neighbor in one direction and a periodic boundary
    iOct = iOct3; // chose an octant which should have a "same size" neighbor in all direction
    for( uint32_t i1=0; i1<blocksize; i1++ )
      for( uint32_t ig=0; ig<ghostWidth; ig++ )
      {
        // Check Bottom border (print top)
        check_cell(iOct,ghostWidth+i1,ig, SMALLER);
        // Check Top border (print bottom)
        check_cell(iOct,ghostWidth+i1,ig+blocksize+ghostWidth, SMALLER);
        // Check Left border
        check_cell(iOct,ig,ghostWidth+i1, SMALLER);
        // Check Right
        check_cell(iOct,ig+blocksize+ghostWidth,ghostWidth+i1, SMALLER);
      }

    for( uint32_t ig1=0; ig1<ghostWidth; ig1++ )
      for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
      {
        //Check Bottom Left
        check_cell(iOct,ig1,ig2, SMALLER);
        //Check Bottom Right
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2, SMALLER);
        //Check Upper Left
        check_cell(iOct,ig1,ig2+blocksize+ghostWidth, SMALLER);
        //Check Upper Right
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth, SMALLER);
      }

    // //Check (full) right border cells
    // for( uint32_t i1=0; i1<blocksize+2*ghostWidth; i1++ )
    //   for( uint32_t ig=0; ig<ghostWidth; ig++ )
    // {
    //   BOOST_CHECK_CLOSE(UgroupHost(ghostWidth+blocksize+ig + (blocksize+2*ghostWidth)*i1, fm[IP], iOct), 
    //                     UgroupHost(ghostWidth+blocksize-1  + (blocksize+2*ghostWidth)*i1, fm[IP], iOct), 0.001);
    // }    
      
  } // end testing CopyFaceBlockCellDataFunctor

} // run_test

} // namespace muscl_block

} // namespace dyablo

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_SUITE(muscl_block)

BOOST_AUTO_TEST_CASE(test_CopyGhostBlockCellData_2D)
{

  run_test(boost::unit_test::framework::master_test_suite().argc,
           boost::unit_test::framework::master_test_suite().argv);

}

BOOST_AUTO_TEST_SUITE_END() /* muscl_block */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */