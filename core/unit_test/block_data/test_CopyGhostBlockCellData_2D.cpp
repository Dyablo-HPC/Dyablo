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
#include "shared/solver_utils.h" // print monitoring information
#include "shared/FieldManager.h"

#ifdef DYABLO_USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

#include "muscl/SolverHydroMuscl.h"
#include "muscl_block/SolverHydroMusclBlock.h"
#include "muscl_block/init/InitImplode.h"

#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyFaceBlockCellData.h"
#include "muscl_block/CopyCornerBlockCellData.h"
#include "muscl_block/CopyGhostBlockCellData.h"
#include "muscl_block/ConvertToPrimitivesHydroFunctor.h"

using Device = Kokkos::DefaultExecutionSpace;

#include <boost/test/unit_test.hpp>
using namespace boost::unit_test;


namespace dyablo
{

namespace muscl_block
{

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

  // anyway create the right solver
  std::unique_ptr<SolverHydroMusclBlock> solver = std::make_unique<SolverHydroMusclBlock>(params, configMap);

  // by now, init condition must have been called

  // just retrieve a field manager
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);

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

  uint32_t nbCellsPerOct_g =
      params.dimType == TWO_D ? bx_g * by_g : bx_g * by_g * bz_g;

  uint32_t nbOctsPerGroup = configMap.getInteger("amr", "nbOctsPerGroup", 32);

  /*
   * allocate/initialize Ugroup
   */

  uint32_t nbOcts = solver->amr_mesh->getNumOctants();
  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";
  

  DataArrayBlock Ugroup = DataArrayBlock("Ugroup", nbCellsPerOct_g, params.nbvar, nbOctsPerGroup);

  std::cout << "Ugroup sizes = " 
              << Ugroup.extent(0) << " "
              << Ugroup.extent(1) << " "
              << Ugroup.extent(2) << "\n";

  uint32_t iGroup = 1;

  FlagArrayBlock Interface_flags = FlagArrayBlock("Interface Flags", nbOctsPerGroup);

  // save solution, just for cross-checking
  solver->save_solution();

  // Define print functions as lambdas
  // Print info about original local (iGroup set in main()) octant
  auto show_octant = [&](uint32_t iOct_local){  
    uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;

    std::cout << "Looking at octant id = " << iOct_global << "\n";
    // octant location
    double x = solver->amr_mesh->getX(iOct_global);
    double y = solver->amr_mesh->getY(iOct_global);
    std::cout << "Octant location : x=" << x << " y=" << y << "\n";
    auto print_neighbor_status = [&]( int codim, int iface)
    { 
      std::vector<uint32_t> iOct_neighbors;
      std::vector<bool> isghost_neighbors;
      solver->amr_mesh->findNeighbours(iOct_global, iface, codim, iOct_neighbors, isghost_neighbors);
      if (iOct_neighbors.size() == 0)
      {
        std::cout << "( no neigh)";
      }
      else if(iOct_neighbors.size() == 1)
      {
        uint32_t neigh_level = isghost_neighbors[0] ? 
                    solver->amr_mesh->getLevel(solver->amr_mesh->getGhostOctant(iOct_neighbors[0])) : 
                    solver->amr_mesh->getLevel(iOct_neighbors[0]);
        if ( solver->amr_mesh->getLevel(iOct_global) > neigh_level )
          std::cout << "( bigger  )";
        else
          std::cout << "(same size)";
      }
      else
      {
        std::cout << "( smaller )";
      }    
    };
    std::cout << "Octant neighbors :" << std::endl;
    print_neighbor_status(2,0) ; print_neighbor_status(1,2) ; print_neighbor_status(2,1); std::cout << std::endl;
    print_neighbor_status(1,0) ; std::cout << "(         )"; print_neighbor_status(1,1); std::cout << std::endl;
    print_neighbor_status(2,2) ; print_neighbor_status(1,3) ; print_neighbor_status(2,3); std::cout << std::endl;
    

    // std::cout << "  FACE_LEFT   "; print_neighbor_status(FACE_LEFT);
    // std::cout << "  FACE_RIGHT  "; print_neighbor_status(FACE_RIGHT);
    // std::cout << "  FACE_TOP    "; print_neighbor_status(FACE_TOP);
    // std::cout << "  FACE_BOTTOM "; print_neighbor_status(FACE_BOTTOM);

    std::cout << "Printing U data from iOct = " << iOct_global << "\n";
    for (uint32_t iz=0; iz<bz; ++iz) 
    {
      for (uint32_t iy=0; iy<by; ++iy)
      {
        for (uint32_t ix=0; ix<bx; ++ix)
        {
          uint32_t index = ix + bx*(iy+by*iz);
          printf("%5f ",solver->U(index,fm[ID],iOct_global));
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
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
        printf("%5f ", Ugroup(index, fm[IP], iOct_local));
      }
      std::cout << "\n";
    } 
  };

  // Octant that have a "same size" neighbor in all direction
  uint32_t iOct1 = 2;
  // Octant that have at least
  // a "larger size" neighbor in one direction
  uint32_t iOct2 = 30;
  // Octant that have have at least
  // an interface with "smaller size" neighbor in one direction and a periodic boundary
  uint32_t iOct3 = 26;

  show_octant(iOct1);
  show_octant(iOct2);
  show_octant(iOct3);  

  // first copy inner cells
  CopyInnerBlockCellDataFunctor::apply(configMap, params, fm, 
                                       blockSizes,
                                       ghostWidth, 
                                       nbOcts,
                                       nbOctsPerGroup,
                                       solver->U, Ugroup, iGroup);

  std::cout << "==========================================";
  std::cout << "Testing CopyFaceBlockCellDataFunctor....\n";
  {
    CopyGhostBlockCellDataFunctor::apply(solver->amr_mesh,
                                        configMap,
                                        params, 
                                        fm,
                                        blockSizes,
                                        ghostWidth,
                                        nbOctsPerGroup,
                                        solver->U, 
                                        solver->Ughost, 
                                        Ugroup, 
                                        iGroup,
                                        solver->interface_flags);
    
    
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
    
    using NEIGH_SIZE = dyablo::muscl_block::CopyFaceBlockCellDataFunctor::NEIGH_SIZE;
    NEIGH_SIZE LARGER = NEIGH_SIZE::NEIGH_IS_LARGER;
    NEIGH_SIZE SMALLER = NEIGH_SIZE::NEIGH_IS_SMALLER;
    // Fetch values from initial conditions and compare to actual value 
    auto check_cell = [&](uint32_t iOct_local, uint32_t ix, uint32_t iy,  NEIGH_SIZE neighbor_size = NEIGH_SIZE::NEIGH_IS_SAME_SIZE )
    { 
      ImplodeParams iParams(configMap);
      uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;
      const real_t octSize = solver->amr_mesh->getSize(iOct_global);
      const real_t cellSize = octSize/bx;
      const real_t x0 = solver->amr_mesh->getNode(iOct_global, 0)[IX];
      const real_t y0 = solver->amr_mesh->getNode(iOct_global, 0)[IY];
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
      real_t& actual = Ugroup(index, fm[IP], iOct_local);

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
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth, LARGER);
      }

    // chose an octant which should have at least
    // a "larger size" neighbor in one direction
    iOct = iOct2;
    for( uint32_t i1=0; i1<blocksize; i1++ )
      for( uint32_t ig=0; ig<ghostWidth; ig++ )
      {
        // Check Bottom border (print top)
        check_cell(iOct,ghostWidth+i1,ig);
        // Check Top border (print bottom)
        check_cell(iOct,ghostWidth+i1,ig+blocksize+ghostWidth,LARGER);
        // Check Left border
        check_cell(iOct,ig,ghostWidth+i1);
        // Check Right
        check_cell(iOct,ig+blocksize+ghostWidth,ghostWidth+i1,LARGER);
      }

    for( uint32_t ig1=0; ig1<ghostWidth; ig1++ )
      for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
      {
        //Check Bottom Left
        check_cell(iOct,ig1,ig2);
        //Check Bottom Right
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2,LARGER);
        //Check Upper Left
        check_cell(iOct,ig1,ig2+blocksize+ghostWidth,LARGER);
        //Check Upper Right
        check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth,LARGER);
      }

    // chose an octant which should have at least
    // an interface with "smaller size" neighbor in one direction and a periodic boundary
    iOct = iOct3; // chose an octant which should have a "same size" neighbor in all direction
    for( uint32_t i1=0; i1<blocksize; i1++ )
      for( uint32_t ig=0; ig<ghostWidth; ig++ )
      {
        // Check Bottom border (print top)
        check_cell(iOct,ghostWidth+i1,ig);
        // Check Top border (print bottom)
        check_cell(iOct,ghostWidth+i1,ig+blocksize+ghostWidth);
        // Check Left border
        check_cell(iOct,ig,ghostWidth+i1, SMALLER);
        // Check Right
        // (border) check_cell(iOct,ig+blocksize+ghostWidth,ghostWidth+i1);
      }

    for( uint32_t ig1=0; ig1<ghostWidth; ig1++ )
      for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
      {
        //Check Bottom Left
        check_cell(iOct,ig1,ig2);
        //Check Bottom Right
        // (border) check_cell(iOct,ig1+blocksize+ghostWidth,ig2);
        //Check Upper Left
        check_cell(iOct,ig1,ig2+blocksize+ghostWidth);
        //Check Upper Right
        // (border) check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth);
      }

    //Check (full) right border cells
    for( uint32_t i1=0; i1<blocksize+2*ghostWidth; i1++ )
      for( uint32_t ig=0; ig<ghostWidth; ig++ )
    {
      BOOST_CHECK_CLOSE(Ugroup(ghostWidth+blocksize+ig + (blocksize+2*ghostWidth)*i1, fm[IP], iOct), 
                        Ugroup(ghostWidth+blocksize-1  + (blocksize+2*ghostWidth)*i1, fm[IP], iOct), 0.001);
    }    
      
  } // end testing CopyFaceBlockCellDataFunctor

} // run_test

} // namespace muscl_block

} // namespace dyablo

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_SUITE(muscl_block)

BOOST_AUTO_TEST_CASE(test_CopyGhostBlockCellData_2D)
{

  run_test(framework::master_test_suite().argc,
           framework::master_test_suite().argv);

}

BOOST_AUTO_TEST_SUITE_END() /* muscl_block */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */

//old main
#if 0
// =======================================================================
// =======================================================================
// =======================================================================
int main(int argc, char *argv[])
{

  // Create MPI session if MPI enabled
#ifdef DYABLO_USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
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

  dyablo::muscl_block::run_test(argc, argv);

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
#endif // old main
