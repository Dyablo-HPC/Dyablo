/**
 * \file test_AMRMetaData.cpp
 * \author Pierre Kestener
 */
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

#ifdef DYABLO_USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

#include "shared/AMRMetaData.h"
#include "muscl_block/LightOctree.h"

#include <iostream>

using Device = Kokkos::DefaultExecutionSpace;

#include <boost/test/unit_test.hpp>

namespace dyablo
{

// ==========================================================
// ==========================================================
template<int dim>
void run_test()
{

  std::cout << "\n\n\n";
  std::cout << "==================================\n";
  std::cout << "AMRMetaData test in dim : " << dim << "\n";
  std::cout << "==================================\n";

  // ================================
  // stage 1 : create a PABLO mesh
  // ================================

  /**<Instantation of a nDimensional pablo uniform object.*/
  bitpit::PabloUniform amr_mesh(dim);

  // Set 2:1 balance
  // codim 1 ==> balance through faces
  // codim 2 ==> balance through faces and corner
  // codim 3 ==> balance through faces, edges and corner (3D only)
  int codim = 1; 
  amr_mesh.setBalanceCodimension(codim);
  
  uint32_t idx=0;
  amr_mesh.setBalance(idx,true);

  /**<set periodic border condition */
  amr_mesh.setPeriodic(0);
  amr_mesh.setPeriodic(1);
  amr_mesh.setPeriodic(2);
  amr_mesh.setPeriodic(3);
  if (dim==3)
  {
    amr_mesh.setPeriodic(4);
    amr_mesh.setPeriodic(5);
  }

  // stage 1
  amr_mesh.adaptGlobalRefine();
  amr_mesh.updateConnectivity();

  // stage 2
  amr_mesh.setMarker(1,1);
  amr_mesh.adapt(true);
  amr_mesh.updateConnectivity();

  // stage 3
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
  /**<(Load)Balance the octree over the processes.*/
  amr_mesh.loadBalance();
#endif

  /* 
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

  std::cout << "Number of octants :" << amr_mesh.getNumOctants() <<  "\n";

  // ============================================================
  // ============================================================

  // print mesh
  if (dim==2) 
  {
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

    // shared_ptr without a deleter
    std::shared_ptr<bitpit::PabloUniform> mesh_ptr(&amr_mesh, [](bitpit::PabloUniform*){});
    // Create LightOctree
    HydroParams params;
    params.level_max = 3;
    muscl_block::LightOctree mesh( mesh_ptr, params );

    // for( int level=0; level<3; level++ )
    // {
    //   std::cout << "level " << level << std::endl;
    //   for( auto p : mesh.oct_map[level] )
    //     std::cout << p.first << ", " << p.second.iOct << std::endl;
    // }

    auto test_oct = [&]( uint32_t ioct, uint32_t first_neighbor[3][3]){
      std::cout << "Octant " << ioct << " :" << std::endl;
      std::cout << "Get Neighbors..." << std::endl;
      Kokkos::View<real_t[3][3]> actual_neighbors("2D::actual_neighbors");
      Kokkos::parallel_for( "find_neighbors", 9, KOKKOS_LAMBDA(int i)
      {
        if(i==4) return; // {0,0,0}
        
        int8_t y = (i/3);
        int8_t x = (i-3*y); 
        muscl_block::LightOctree::offset_t offset{(int8_t)(x-1),(int8_t)(y-1),0};

        muscl_block::LightOctree::NeighborList ns = mesh.findNeighbors( {ioct,false}, offset );
        actual_neighbors(y,x) = ns[0].iOct;
      } );
      std::cout << "[DONE]" << std::endl;
      Kokkos::View<real_t[3][3]>::HostMirror actual_neighbors_host("2D::actual_neighbors_host");
      Kokkos::deep_copy(actual_neighbors_host, actual_neighbors);

      std::cout << "expected" << std::endl;
      for(int8_t y=2; y>=0; y--)
      {
        for(int8_t x=0; x<3; x++)
        {
          if( x==1 && y==1 ) 
          {
            std::cout <<  std::setw(5) << ioct;
            continue;
          }
          uint32_t expected_neighbor = first_neighbor[2-y][x];
          std::cout << std::setw(5) << expected_neighbor;
        }
        std::cout << std::endl;
      }
      std::cout << "actual" << std::endl;
      for(int8_t y=2; y>=0; y--)
      {
        for(int8_t x=0; x<3; x++)
        {
          if( x==1 && y==1 ) 
          {
            std::cout <<  std::setw(5) << ioct;
            continue;
          }
          uint32_t actual_neighbor = actual_neighbors_host(y,x);

          std::cout << std::setw(5) << actual_neighbor;
        }
        std::cout << std::endl;
      }      
      for(int8_t y=2; y>=0; y--)
      {
        for(int8_t x=0; x<3; x++)
        {
          if( x==1 && y==1 ) 
          {
            continue;
          }
          uint32_t expected_neighbor = first_neighbor[2-y][x];
          uint32_t actual_neighbor = actual_neighbors_host(y,x);

          BOOST_CHECK_EQUAL( actual_neighbor, expected_neighbor);
        }
      }
    };

    // uint32_t neighbors_8[3][3] = {{11,12,12},
    //                               { 3, 0, 9},
    //                               { 3, 6, 7}};
    // test_oct( 8, neighbors_8 );

    uint32_t neighbors_7[3][3] = {{ 8, 9,10},
                                  { 6, 0,10},
                                  { 4, 4, 5}};
    test_oct( 7, neighbors_7 );

    uint32_t neighbors_5[3][3] = {{ 7,10, 2},
                                  { 4, 0, 0},
                                  {14,15,11}};
    test_oct( 5, neighbors_5 );
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
    std::cout << "   |  0  |  1  |  8  |  9  |  /            ^       | 9  | 11 | 25 | 27 |    \n";
    std::cout << "   |_____|_____|_____|_____| /             |z      |____|____|____|____|    \n";
    std::cout << "                                                                            \n";
    std::cout << "  Front Half : \n";
    std::cout << "    ___________ _____ _____     ___________ _____ _____     ___________ _____ _____  \n";
    std::cout << "   |           |     |     |   |           |     |     |   |           |     |     | \n";
    std::cout << "   |           | 37  | 38  |   |           | 37  | 38  |   |           | 39  | 40  | \n";
    std::cout << "   |    32     |-----|-----|   |    32     |-----|-----|   |    32     |-----|-----| \n";
    std::cout << "   |           | 33  | 34  |   |           | 33  | 34  |   |           | 35  | 36  | \n";
    std::cout << "   |___________|_____|_____|   |___________|_____|_____|   |___________|_____|_____| \n";
    std::cout << "   |     |     |16|17|     |   |     |     |18|19|     |   |     |     |     |     | \n";
    std::cout << "   |  4  |  5  |-----| 20  |   |  4  |  5  |-----| 20  |   |  6  |  7  |  21 | 22  | \n";
    std::cout << "   |_____|_____|12|13|_____|   |_____|_____|14|15|_____|   |-----|-----|-----|-----| \n";
    std::cout << "   |     |     |     |     |   |     |     |     |     |   |     |     |     |     | \n";
    std::cout << "   |  0  |  1  |  8  |  9  |   |  0  |  1  |  8  |  9  |   |  2  |  3  |  10 | 11  | \n";
    std::cout << "   |_____|_____|_____|_____|   |_____|_____|_____|_____|   |_____|_____|_____|_____| \n";
    std::cout << "\n";
    std::cout << "  Back Half \n";
    std::cout << "    ___________ ___________        ___________ ___________     \n";
    std::cout << "   |           |           |      |           |           |    \n";
    std::cout << "   |           |           |      |           |           |    \n";
    std::cout << "   |    41     |    42     |      |    41     |    42     |    \n";
    std::cout << "   |           |           |      |           |           |    \n";
    std::cout << "   |___________|_____ _____|      |___________|_____ _____|    \n";
    std::cout << "   |           |     |     |      |           |     |     |    \n";
    std::cout << "   |           |  28 | 29  |      |           |  30 | 31  |    \n";
    std::cout << "   |    23     |-----|-----|      |    23     |-----|-----|    \n";
    std::cout << "   |           |     |     |      |           |     |     |    \n";
    std::cout << "   |           |  24 | 25  |      |           |  26 | 27  |    \n";
    std::cout << "   |___________|_____|_____|      |___________|_____|_____|    \n";


    // Create LightOctree
    std::shared_ptr<bitpit::PabloUniform> mesh_ptr(&amr_mesh, [](bitpit::PabloUniform*){});
    HydroParams params;
    params.level_max = 3;
    muscl_block::LightOctree mesh( mesh_ptr, params );

    auto test_oct = [&]( uint32_t ioct, uint32_t first_neighbor[3][3][3]){
      std::cout << "Octant " << ioct << " :" << std::endl;
      std::cout << "Get Neighbors..." << std::endl;
      Kokkos::View<real_t[3][3][3]> actual_neighbors("3D::actual_neighbors");
      Kokkos::parallel_for( "find_neighbors", 3*3*3, KOKKOS_LAMBDA(int i)
      {
        if(i==13) return; // {0,0,0}
        
        // i = x + 3*y + 3*3*z
        int8_t z = (i/(3*3));
        int8_t y = (i-z*3*3)/3;
        int8_t x = (i-3*y-3*3*z); 
        muscl_block::LightOctree::offset_t offset{(int8_t)(x-1),(int8_t)(y-1),(int8_t)(z-1)};

        muscl_block::LightOctree::NeighborList ns = mesh.findNeighbors( {ioct,false}, offset );
        actual_neighbors(z,y,x) = ns[0].iOct;
      } );
      std::cout << "[DONE]" << std::endl;
      Kokkos::View<real_t[3][3][3]>::HostMirror actual_neighbors_host("2D::actual_neighbors_host");
      Kokkos::deep_copy(actual_neighbors_host, actual_neighbors);

      std::cout << "expected" << std::endl;
      for(int8_t y=2; y>=0; y--)
      {
        for(int8_t z=2; z>=0; z--)
        {
          for(int8_t x=0; x<3; x++)
          {
            if( x==1 && y==1 && z==1 ) 
            {
              std::cout <<  std::setw(5) << ioct;
              continue;
            }
            uint32_t expected_neighbor = first_neighbor[2-y][2-z][x];
            std::cout << std::setw(5) << expected_neighbor;
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
      std::cout << "actual" << std::endl;
      for(int8_t y=2; y>=0; y--)
      {
        for(int8_t z=2; z>=0; z--)
        {
          for(int8_t x=0; x<3; x++)
          {
            if( x==1 && y==1 && z==1 ) 
            {
              std::cout <<  std::setw(5) << ioct;
              continue;
            }
            uint32_t actual_neighbor = actual_neighbors_host(z,y,x);

            std::cout << std::setw(5) << actual_neighbor;
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      } 
      // Test equal     
      for(int8_t y=2; y>=0; y--)
      {
        for(int8_t z=2; z>=0; z--)
        {
          for(int8_t x=0; x<3; x++)
          {
            if( x==1 && y==1 && z==1 ) 
            {
              continue;
            }
            uint32_t expected_neighbor = first_neighbor[2-y][2-z][x];
            uint32_t actual_neighbor = actual_neighbors_host(z,y,x);

            BOOST_CHECK_EQUAL( actual_neighbor, expected_neighbor);
          }
        }
      }
    };

    uint32_t neighbors_5[3][3][3] = {
     {{32   ,32   ,35},
      {6    ,7    ,21},
      {2    ,3    ,10}},
     {{32   ,32   ,33},
      {4    ,5    ,12},
      {0    ,1    ,8 }},
     {{41   ,41   ,42},
      {23   ,23   ,30},
      {23   ,23   ,26}}
    };
    test_oct( 5, neighbors_5 );
  }
} // run_test

} // dyablo



BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_CASE(test_LightOctree)
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

