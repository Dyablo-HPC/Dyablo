/**
 * \file test_CopyGhostBlockCellData_3D.cpp
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
  std::string input_file = argc>1 ? std::string(argv[1]) : "./block_data/test_implode_3D_block.ini";
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
  uint32_t bz = configMap.getInteger("amr", "bz", 0);;

  // block sizes with ghosts
  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = by + 2 * ghostWidth;
  uint32_t bz_g = bz + 2 * ghostWidth;;

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
    double z = solver->amr_mesh->getZ(iOct_global);
    std::cout << "Octant location : x=" << x << " y=" << y << " z=" << z << "\n";
    auto print_neighbor_status = [&]( int x, int y, int z)
    { 
      constexpr uint8_t iface_from_pos[3][3][3] = {
                            {{ 1, 3, 2 },
                             { 1, 5, 2 },
                             { 3, 4, 4 }},
                            {{ 5, 3, 6 },
                             { 1, 0, 2 },
                             { 7, 4, 8 }},
                            {{ 5, 11, 6 },
                             { 9, 6, 10 },
                             { 7, 12, 8 }}};
      constexpr uint8_t codim_from_pos[3][3][3] = {
                            {{ 3, 2, 3 },
                             { 2, 1, 2 },
                             { 3, 2, 3 }},
                            {{ 2, 1, 2 },
                             { 1, 0, 1 },
                             { 2, 1, 2 }},
                            {{ 3, 2, 3 },
                             { 2, 1, 2 },
                             { 3, 2, 3 }}};

      int iface = iface_from_pos[z][y][x]-1;
      int codim = codim_from_pos[z][y][x];

      std::vector<uint32_t> iOct_neighbors;
      std::vector<bool> isghost_neighbors;
      solver->amr_mesh->findNeighbours(iOct_global, iface, codim, iOct_neighbors, isghost_neighbors);
      if (iOct_neighbors.size() == 0)
      {
        std::cout <<    "( nothing ) ";
      }
      else if(iOct_neighbors.size() == 1)
      {
        uint32_t neigh_level = isghost_neighbors[0] ? 
                    solver->amr_mesh->getLevel(solver->amr_mesh->getGhostOctant(iOct_neighbors[0])) : 
                    solver->amr_mesh->getLevel(iOct_neighbors[0]);
        if ( solver->amr_mesh->getLevel(iOct_global) > neigh_level )
          std::cout << "( bigger  ) ";
        else
          std::cout << "(same size) ";
      }
      else
        std::cout << "( smaller ) ";  
    };
    std::cout << "Octant neighbors :" << std::endl;
    
    for (uint32_t iz=0; iz<3; ++iz) 
    {
      for (uint32_t iy=0; iy<3; ++iy)
      {
        for (uint32_t ix=0; ix<3; ++ix)
        {
          print_neighbor_status( ix, iy, iz );
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

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
    for (uint32_t iz=0; iz < bz_g; ++iz) 
    {
      for (uint32_t iy = 0; iy < by_g; ++iy)
      {
        for (uint32_t ix = 0; ix < bx_g; ++ix)
        {
          uint32_t index = ix + bx_g*(iy+by_g*iz);
          printf("%5f ", Ugroup(index, fm[IP], iOct_local));
        }
        std::cout << "\n";
      } 
      std::cout << "\n";
    }
  };


  // Octant that have a "same size" neighbor in all direction
  uint32_t iOct1 = 11;
  // Octant that have at least
  // a "larger size" neighbor in one direction
  uint32_t iOct2 = 27;
  // Octant that have have at least
  // an interface with "smaller size" neighbor in one direction and a periodic boundary
  uint32_t iOct3 = 28;

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
                                        Interface_flags);   
  // CopyFaceBlockCellDataFunctor::apply(solver->amr_mesh,
  //                                       configMap,
  //                                       params, 
  //                                       fm,
  //                                       blockSizes,
  //                                       ghostWidth,
  //                                       nbOctsPerGroup,
  //                                       solver->U, 
  //                                       solver->Ughost, 
  //                                       Ugroup, 
  //                                       iGroup,
  //                                       Interface_flags);    
    
    using NEIGH_SIZE = dyablo::muscl_block::CopyFaceBlockCellDataFunctor::NEIGH_SIZE;
    NEIGH_SIZE LARGER = NEIGH_SIZE::NEIGH_IS_LARGER;
    NEIGH_SIZE SMALLER = NEIGH_SIZE::NEIGH_IS_SMALLER;
    // Fetch values from initial conditions and compare to actual value 
    auto check_cell = [&](uint32_t iOct_local, uint32_t ix, uint32_t iy, uint32_t iz, NEIGH_SIZE neighbor_size = NEIGH_SIZE::NEIGH_IS_SAME_SIZE )
    { 
      ImplodeParams iParams(configMap);
      uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;
      const real_t octSize = solver->amr_mesh->getSize(iOct_global);
      const real_t cellSize = octSize/bx;
      const real_t x0 = solver->amr_mesh->getNode(iOct_global, 0)[IX];
      const real_t y0 = solver->amr_mesh->getNode(iOct_global, 0)[IY];
      const real_t z0 = solver->amr_mesh->getNode(iOct_global, 0)[IZ];
      real_t x = x0 + ix*cellSize - ghostWidth*cellSize + cellSize/2;
      real_t y = y0 + iy*cellSize - ghostWidth*cellSize + cellSize/2;
      real_t z = z0 + iz*cellSize - ghostWidth*cellSize + cellSize/2;

      real_t expected;
      if( neighbor_size == LARGER )
      {
        // Works because ghostWidth and blockSizes are pair
        uint32_t ix_larger = (ix/2)*2 + 1;
        uint32_t iy_larger = (iy/2)*2 + 1;
        uint32_t iz_larger = (iz/2)*2 + 1;
        real_t x_larger = x0 + ix_larger*cellSize - ghostWidth*cellSize;
        real_t y_larger = y0 + iy_larger*cellSize - ghostWidth*cellSize;
        real_t z_larger = z0 + iz_larger*cellSize - ghostWidth*cellSize;
        expected = InitImplodeDataFunctor::value( params, iParams, x_larger,y_larger,z_larger )[IP];
      }
      else if( neighbor_size == SMALLER )
      {
        expected = 0;        
        for(int dz=0; dz<2; dz++)
          for(int dy=0; dy<2; dy++)
            for(int dx=0; dx<2; dx++)
            {
              real_t x_smaller = x - cellSize/4 + dx*(cellSize/2);
              real_t y_smaller = y - cellSize/4 + dy*(cellSize/2);
              real_t z_smaller = z - cellSize/4 + dz*(cellSize/2);
              expected += InitImplodeDataFunctor::value( params, iParams, x_smaller ,y_smaller ,z_smaller  )[IP];
            }
        expected = expected/8; 
      }
      else
      {
        expected = InitImplodeDataFunctor::value( params, iParams, x,y,z )[IP];
      }
      
      uint32_t index = ix + bx_g * iy + bx_g * by_g * iz;
      real_t& actual = Ugroup(index, fm[IP], iOct_local);

      BOOST_CHECK_CLOSE(actual, expected, 0.0001);      
      
      //// ---------------------- Debug prints ------------------------
      if( actual != expected )
      {
        std::cout << "Value at iOct : " << iOct_global << " pos ghosted : "
                    "(" << ix << "," << iy << "," << iz << ") = "
                    "(" << x << "," << y << "," << z << ") :" << std::endl;
        std::cout << "  Should be (implode) : " <<  expected << std::endl;
        std::cout << "  Is                  : " <<  actual << std::endl;
      }
      //actual = 9.1111; //To visualize which cells have been tested
      //actual = (actual==expected) ? 9.1111 : 9.2222; //To visualize which tested cells are wrong
      //actual = actual - expected; //To visualize difference to expected
      //// ---------------------- Debug prints ------------------------
    };

    { // Test values for iOct1
      uint32_t iOct = iOct1;

      //Check faces
      for( uint32_t ig=0; ig<ghostWidth; ig++)
        for( uint32_t i1=ghostWidth; i1<ghostWidth+bx; i1++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        check_cell(iOct,ig,i1,i2);                 //Left
        check_cell(iOct,bx+ghostWidth+ig,i1,i2);   //Right
        check_cell(iOct,i1,ig,i2);                 // Bottom (print up)
        check_cell(iOct,i1,ig+ghostWidth+by,i2);   // Top (print down)
        check_cell(iOct,i1,i2,ig);                 // front 
        check_cell(iOct,i1,i2,ig+ghostWidth+bz);   // back
      }
      //Check edges
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        //Edges in X direction
        check_cell( iOct, i2, ig1, ig2 );
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2 );
        check_cell( iOct, i2, ig1, ig2+bx+ghostWidth );
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2+bx+ghostWidth, LARGER );
        //Edges in Y direction
        check_cell( iOct, ig1, i2, ig2 );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2 );
        check_cell( iOct, ig1, i2, ig2+bx+ghostWidth );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2+bx+ghostWidth, LARGER );
        //Edges in Z direction
        check_cell( iOct, ig1, ig2, i2 );
        check_cell( iOct, ig1+bx+ghostWidth, ig2, i2 );
        check_cell( iOct, ig1, ig2+bx+ghostWidth, i2 );
        check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, i2, LARGER );
      }

      //Check corners
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t ig3=0; ig3<ghostWidth; ig3++ )
      {
          check_cell( iOct, ig1, ig2, ig3 );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3 );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3 );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3, LARGER );
          check_cell( iOct, ig1, ig2, ig3+bx+ghostWidth );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3+bx+ghostWidth, LARGER );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3+bx+ghostWidth, LARGER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3+bx+ghostWidth, LARGER );
      }

    }

    { // Test values for iOct2
      uint32_t iOct = iOct2;

      //Check faces
      for( uint32_t ig=0; ig<ghostWidth; ig++)
        for( uint32_t i1=ghostWidth; i1<ghostWidth+bx; i1++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        check_cell(iOct,ig,i1,i2);                 //Left
        check_cell(iOct,bx+ghostWidth+ig,i1,i2,LARGER);   //Right
        check_cell(iOct,i1,ig,i2);                 // Bottom (print up)
        check_cell(iOct,i1,ig+ghostWidth+by,i2,LARGER);   // Top (print down)
        check_cell(iOct,i1,i2,ig);                 // front 
        check_cell(iOct,i1,i2,ig+ghostWidth+bz,LARGER);   // back
      }
      //Check edges
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        //Edges in X direction
        check_cell( iOct, i2, ig1, ig2 );
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2,LARGER );
        check_cell( iOct, i2, ig1, ig2+bx+ghostWidth, LARGER);
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2+bx+ghostWidth, LARGER );
        //Edges in Y direction
        check_cell( iOct, ig1, i2, ig2 );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2,LARGER );
        check_cell( iOct, ig1, i2, ig2+bx+ghostWidth, LARGER );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2+bx+ghostWidth, LARGER );
        //Edges in Z direction
        check_cell( iOct, ig1, ig2, i2 );
        check_cell( iOct, ig1+bx+ghostWidth, ig2, i2,LARGER );
        check_cell( iOct, ig1, ig2+bx+ghostWidth, i2,LARGER );
        check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, i2, LARGER );
      }

      //Check corners
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t ig3=0; ig3<ghostWidth; ig3++ )
      {
          check_cell( iOct, ig1, ig2, ig3 );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3,LARGER );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3,LARGER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3, LARGER );
          check_cell( iOct, ig1, ig2, ig3+bx+ghostWidth,LARGER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3+bx+ghostWidth, LARGER );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3+bx+ghostWidth, LARGER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3+bx+ghostWidth, LARGER );
      }

    }

    { // Test values for iOct3
      uint32_t iOct = iOct3;

      //Check faces
      for( uint32_t ig=0; ig<ghostWidth; ig++)
        for( uint32_t i1=ghostWidth; i1<ghostWidth+bx; i1++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        check_cell(iOct,ig,i1,i2,SMALLER);                 //Left
        check_cell(iOct,bx+ghostWidth+ig,i1,i2);   //Right
        check_cell(iOct,i1,ig,i2,SMALLER);                 // Bottom (print up)
        check_cell(iOct,i1,ig+ghostWidth+by,i2);   // Top (print down)
        //check_cell(iOct,i1,i2,ig);                 // front 
        check_cell(iOct,i1,i2,ig+ghostWidth+bz);   // back
      }
      //Check edges
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        //Edges in X direction
        //check_cell( iOct, i2, ig1, ig2 );
        //check_cell( iOct, i2, ig1+bx+ghostWidth, ig2 );
        check_cell( iOct, i2, ig1, ig2+bx+ghostWidth );
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2+bx+ghostWidth );
        //Edges in Y direction
        //check_cell( iOct, ig1, i2, ig2 );
        //check_cell( iOct, ig1+bx+ghostWidth, i2, ig2 );
        check_cell( iOct, ig1, i2, ig2+bx+ghostWidth );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2+bx+ghostWidth );
        //Edges in Z direction
        check_cell( iOct, ig1, ig2, i2, SMALLER );
        check_cell( iOct, ig1+bx+ghostWidth, ig2, i2 );
        check_cell( iOct, ig1, ig2+bx+ghostWidth, i2 );
        check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, i2, LARGER );
      }

      //Check corners
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t ig3=0; ig3<ghostWidth; ig3++ )
      {
          //check_cell( iOct, ig1, ig2, ig3 );
          //check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3 );
          //check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3 );
          //check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3, LARGER );
          check_cell( iOct, ig1, ig2, ig3+bx+ghostWidth );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3+bx+ghostWidth );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3+bx+ghostWidth );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3+bx+ghostWidth, LARGER );
      }

      //Check border on Z
      for( uint32_t ig=0; ig<ghostWidth; ig++)
        for( uint32_t i1=0; i1<2*ghostWidth+bx; i1++ )
          for( uint32_t i2=0; i2<2*ghostWidth+bx; i2++ )
      {
          BOOST_CHECK_CLOSE(Ugroup(i1 + (bx+2*ghostWidth)*(i2 + (bx+2*ghostWidth)*ig), fm[IP], iOct), 
                            Ugroup(i1 + (bx+2*ghostWidth)*(i2 + (bx+2*ghostWidth)*ghostWidth), fm[IP], iOct), 0.001);
      }

    }


    show_octant_group(iOct1);
    show_octant_group(iOct2);
    show_octant_group(iOct3);     
      
  } // end testing CopyFaceBlockCellDataFunctor

} // run_test

} // namespace muscl_block

} // namespace dyablo

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_SUITE(muscl_block)

BOOST_AUTO_TEST_CASE(test_CopyGhostBlockCellData_3D)
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
