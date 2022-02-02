/**
 * \file test_AMRMetaData.cpp
 * \author Pierre Kestener
 */
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

#include "utils/monitoring/SimpleTimer.h"
#include "amr/LightOctree.h"
#include "legacy/HydroParams.h"

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
  // Set 2:1 balance
  // codim 1 ==> balance through faces
  // codim 2 ==> balance through faces and corner
  // codim 3 ==> balance through faces, edges and corner (3D only)
  HydroParams params{};
  params.level_max = 3;
  params.level_min = 0;
  int codim = dim; 
  dyablo::AMRmesh amr_mesh(dim, codim, {true,true,true}, params.level_min, params.level_max);

  // stage 1
  amr_mesh.adaptGlobalRefine();

  // stage 2
  amr_mesh.setMarker(1,1);
  amr_mesh.adapt(true);

  // stage 3
  if (dim==2) 
  {
    amr_mesh.setMarker(3,1);
    amr_mesh.adapt(true);
  }
  else
  {
    amr_mesh.setMarker(5,1);
    amr_mesh.adapt(true);
  }
  std::cout << "Mesh size is " << amr_mesh.getNumOctants() << "\n";

  /**<(Load)Balance the octree over the processes.*/
  amr_mesh.loadBalance();

  /* 
   * 2d mesh should be exactly like this : 19 octants
   *   ___________ __________ 
   *  |     |     |     |    |
   *  |  13 | 14  |  17 | 18 |
   *  |-----+-----|-----+----|
   *  |  11 | 12  |  15 | 16 |
   *  |_____|____ |_____|____|
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
   * THIS IS NOT UP TO DATE
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
    std::cout << "  |     |     |     |    |    \n";
    std::cout << "  |  13 | 14  |  17 | 18 |    \n";
    std::cout << "  |-----+-----|-----+----|    \n";
    std::cout << "  |  11 | 12  |  15 | 16 |    \n";
    std::cout << "  |_____|____ |_____|____|    \n";
    std::cout << "  |     |     | 8|9 |    |    \n";
    std::cout << "  |  2  |  3  | 6|7 | 10 |    \n";
    std::cout << "  |-----+-----|-----+----|    \n";
    std::cout << "  |  0  |  1  |  4  | 5  |    \n";
    std::cout << "  |_____|_____|_____|____|    \n";
    std::cout << "                              \n";

    // Create LightOctree
    const LightOctree& mesh = amr_mesh.getLightOctree();

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
        LightOctree::offset_t offset{(int8_t)(x-1),(int8_t)(y-1),0};

        LightOctree::NeighborList ns = mesh.findNeighbors( {ioct,false}, offset );
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

    Kokkos::View<real_t[8]> octant_data("octant_data");
    Kokkos::parallel_for( "get_octant_data", 1, KOKKOS_LAMBDA(int i)
    {
      octant_data(0) = mesh.getCorner({10,false})[IX];
      octant_data(1) = mesh.getCorner({10,false})[IY];
      octant_data(2) = mesh.getCorner({10,false})[IZ];
      octant_data(3) = mesh.getCenter({10,false})[IX];
      octant_data(4) = mesh.getCenter({10,false})[IY];
      octant_data(5) = mesh.getCenter({10,false})[IZ];
      octant_data(6) = mesh.getSize({10,false});
      octant_data(7) = mesh.getLevel({10,false});
    });
    auto octant_data_host = Kokkos::create_mirror_view(octant_data);
    Kokkos::deep_copy(octant_data_host, octant_data);

    BOOST_CHECK_CLOSE(octant_data_host(0) , 0.75, 0.0001 );
    BOOST_CHECK_CLOSE(octant_data_host(1) , 0.25, 0.0001  );
    BOOST_CHECK_CLOSE(octant_data_host(2) , 0.0, 0.0001  );
    BOOST_CHECK_CLOSE(octant_data_host(3) , 0.875, 0.0001 );
    BOOST_CHECK_CLOSE(octant_data_host(4) , 0.375, 0.0001  );
    BOOST_CHECK_CLOSE(octant_data_host(5) , 0.0, 0.0001  );
    BOOST_CHECK_CLOSE(octant_data_host(6) , 0.25, 0.0001  );
    BOOST_CHECK_EQUAL(octant_data_host(7) , 2  );

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
                                  {17,18,13}};
    test_oct( 5, neighbors_5 );

  } 
  else 
  {
    // THIS IS NOT UP TO DATE
    // std::cout << "            ________________________                                        \n";
    // std::cout << "           /           /           /|                                       \n";
    // std::cout << "          /           /           / |                                       \n";
    // std::cout << "         /   41      /    42     /  |                                       \n";
    // std::cout << "        /           /           /   |                                       \n";
    // std::cout << "       /___________/___________/    |                                       \n";
    // std::cout << "      /           / 39  / 40  / |   |                                       \n";
    // std::cout << "     /   32      /____ /_____/  |   |                                       \n";
    // std::cout << "    /           /  37 / 38  / | |   |               ___________________     \n";
    // std::cout << "   /___________/_____/_____/  |/|  /|              |    |    |         |    \n";
    // std::cout << "   |           |     |     | /| | / |              | 38 | 40 |         |    \n";
    // std::cout << "   |           | 37  | 38  |/ | |/| |              |____|____|    42   |    \n";
    // std::cout << "   |    32     |-----|-----|  |/| |/|              |    |    |         |    \n";
    // std::cout << "   |           | 33  | 34  |  / | | |      <====   | 34 | 36 |         |    \n";
    // std::cout << "   |___________|_____|_____| /| |/ |/              |____|____|_________|    \n";
    // std::cout << "   |     |     |16|17|     |  | | /                |    |    |    |    |    \n";
    // std::cout << "   |  4  |  5  |-----| 20  |  / |/                 | 20 | 22 | 29 | 31 |    \n";
    // std::cout << "   |_____|_____|12|13|_____| /| /                  |____|____|____|____|    \n";
    // std::cout << "   |     |     |     |     |  |/                   |    |    |    |    |    \n";
    // std::cout << "   |  0  |  1  |  8  |  9  |  /            ^       | 9  | 11 | 25 | 27 |    \n";
    // std::cout << "   |_____|_____|_____|_____| /             |z      |____|____|____|____|    \n";
    // std::cout << "                                                                            \n";
    // std::cout << "  Front Half : \n";
    // std::cout << "    ___________ _____ _____     ___________ _____ _____     ___________ _____ _____  \n";
    // std::cout << "   |           |     |     |   |           |     |     |   |           |     |     | \n";
    // std::cout << "   |           | 37  | 38  |   |           | 37  | 38  |   |           | 39  | 40  | \n";
    // std::cout << "   |    32     |-----|-----|   |    32     |-----|-----|   |    32     |-----|-----| \n";
    // std::cout << "   |           | 33  | 34  |   |           | 33  | 34  |   |           | 35  | 36  | \n";
    // std::cout << "   |___________|_____|_____|   |___________|_____|_____|   |___________|_____|_____| \n";
    // std::cout << "   |     |     |16|17|     |   |     |     |18|19|     |   |     |     |     |     | \n";
    // std::cout << "   |  4  |  5  |-----| 20  |   |  4  |  5  |-----| 20  |   |  6  |  7  |  21 | 22  | \n";
    // std::cout << "   |_____|_____|12|13|_____|   |_____|_____|14|15|_____|   |-----|-----|-----|-----| \n";
    // std::cout << "   |     |     |     |     |   |     |     |     |     |   |     |     |     |     | \n";
    // std::cout << "   |  0  |  1  |  8  |  9  |   |  0  |  1  |  8  |  9  |   |  2  |  3  |  10 | 11  | \n";
    // std::cout << "   |_____|_____|_____|_____|   |_____|_____|_____|_____|   |_____|_____|_____|_____| \n";
    // std::cout << "\n";
    // std::cout << "  Back Half \n";
    // std::cout << "    ___________ ___________        ___________ ___________     \n";
    // std::cout << "   |           |           |      |           |           |    \n";
    // std::cout << "   |           |           |      |           |           |    \n";
    // std::cout << "   |    41     |    42     |      |    41     |    42     |    \n";
    // std::cout << "   |           |           |      |           |           |    \n";
    // std::cout << "   |___________|_____ _____|      |___________|_____ _____|    \n";
    // std::cout << "   |           |     |     |      |           |     |     |    \n";
    // std::cout << "   |           |  28 | 29  |      |           |  30 | 31  |    \n";
    // std::cout << "   |    23     |-----|-----|      |    23     |-----|-----|    \n";
    // std::cout << "   |           |     |     |      |           |     |     |    \n";
    // std::cout << "   |           |  24 | 25  |      |           |  26 | 27  |    \n";
    // std::cout << "   |___________|_____|_____|      |___________|_____|_____|    \n";


    // Create LightOctree
    const LightOctree& mesh = amr_mesh.getLightOctree();

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
        LightOctree::offset_t offset{(int8_t)(x-1),(int8_t)(y-1),(int8_t)(z-1)};

        LightOctree::NeighborList ns = mesh.findNeighbors( {ioct,false}, offset );
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

    Kokkos::View<real_t[8]> octant_data("octant_data");
    Kokkos::parallel_for( "get_octant_data", 1, KOKKOS_LAMBDA(int i)
    {
      octant_data(0) = mesh.getCorner({18,false})[IX];
      octant_data(1) = mesh.getCorner({18,false})[IY];
      octant_data(2) = mesh.getCorner({18,false})[IZ];
      octant_data(3) = mesh.getCenter({18,false})[IX];
      octant_data(4) = mesh.getCenter({18,false})[IY];
      octant_data(5) = mesh.getCenter({18,false})[IZ];
      octant_data(6) = mesh.getSize({18,false});
      octant_data(7) = mesh.getLevel({18,false});
    });
    auto octant_data_host = Kokkos::create_mirror_view(octant_data);
    Kokkos::deep_copy(octant_data_host, octant_data);

    BOOST_CHECK_CLOSE(octant_data_host(0) , 0.5, 0.0001 );
    BOOST_CHECK_CLOSE(octant_data_host(1) , 0.125, 0.0001  );
    BOOST_CHECK_CLOSE(octant_data_host(2) , 0.375, 0.0001  );
    BOOST_CHECK_CLOSE(octant_data_host(3) , 0.5625, 0.0001 );
    BOOST_CHECK_CLOSE(octant_data_host(4) , 0.1875, 0.0001  );
    BOOST_CHECK_CLOSE(octant_data_host(5) , 0.4375, 0.0001  );
    BOOST_CHECK_CLOSE(octant_data_host(6) , 0.125, 0.0001  );
    BOOST_CHECK_EQUAL(octant_data_host(7) , 3  );

    uint32_t neighbors_5[3][3][3] = {
     {{41   ,42   ,49},
      {6    ,7    ,21},
      {2    ,3    ,10}},
     {{39   ,40   ,47},
      {4    ,5    ,12},
      {0    ,1    ,8 }},
     {{57   ,58   ,65},
      {29   ,30   ,37},
      {25   ,26   ,33}}
    };
    test_oct( 5, neighbors_5 );
  }
} // run_test

} // dyablo

//template< typename LightOctree_t >
void test_perf()
{
  constexpr int level_min = 5; //! Min AMR level
  constexpr int level_max = 8; //! Max AMR level
  constexpr int ndim = 3; //! 3D
  // Refine cells at distance between min and max from center of box
  constexpr real_t refine_circle_radius_min = 0.20; 
  constexpr real_t refine_circle_radius_max = 0.25;

  std::cout << "\n\n\n";
  std::cout << "==================================\n";
  std::cout << "   LightOctree perf \n";
  std::cout << "==================================\n";

  std::cout << "Setup PABLO mesh ..." << std::endl;
  //uint8_t CODIM_FACE = 1;
  //uint8_t CODIM_EDGE = 2;
  int CODIM_CORNER = 3;
  dyablo::AMRmesh amr_mesh(ndim, CODIM_CORNER, {true,true,true}, level_min, level_max);
  {
    // Global refine until level_min
    for(int i=0; i<level_min; i++)
      amr_mesh.adaptGlobalRefine();
    // Refine over a circle until level_max
    for(int level=level_min; level<level_max; level++)
    {
      for(uint32_t iOct=0; iOct<amr_mesh.getNumOctants(); iOct++)
      {
        auto center = amr_mesh.getCenter(iOct);
        center[IX] -= 0.5;
        center[IY] -= 0.5;
        center[IZ] -= 0.5;
        real_t d2 = center[IX]*center[IX] + center[IY]*center[IY] + center[IZ]*center[IZ];

        real_t rmin2 = refine_circle_radius_min*refine_circle_radius_min;
        real_t rmax2 = refine_circle_radius_max*refine_circle_radius_max;
        if( rmin2 <= d2 && d2 <= rmax2 )
        {
          amr_mesh.setMarker(iOct, 1);
        }
      }
      amr_mesh.adapt();
    }
  }
  uint64_t nbOct = amr_mesh.getNumOctants();
  std::cout << " Octant count : " << nbOct << std::endl;

  using LightOctree_t = dyablo::LightOctree;

  std::cout << "Construct LightOctree..." << std::endl;
  HydroParams params;
  params.level_max = level_max;

  SimpleTimer time_lmesh_construct;
  const LightOctree_t& lmesh = amr_mesh.getLightOctree();
  Kokkos::fence();
  time_lmesh_construct.stop();
  std::cout << "Done in " << time_lmesh_construct.elapsed()*1000 << " ms (cpu after fence)" << std::endl;

  std::cout << "Measure LightOctree access performance..." << std::endl;
  {
    Kokkos::View<uint32_t*[5]> neighbors_view("neighbors", nbOct); //(size,n1,n2,n3,n4)

    SimpleTimer time_lmesh_findneighbors;
    Kokkos::parallel_for( "LightOctree_access_neighbors", nbOct,
                          KOKKOS_LAMBDA( uint32_t iOct ){
      for(int i=0; i<3*3*3; i++)
      {
        int8_t nz = i/(3*3);
        int8_t ny = (i-nz*3*3)/3;
        int8_t nx = i-ny*3-nz*3*3;
        LightOctree_t::offset_t neigh = {(int8_t)(nx-1),(int8_t)(ny-1),(int8_t)(nz-1)};

        if(neigh[IX]==0 && neigh[IY]==0 && neigh[IZ]==0) return;

        LightOctree_t::NeighborList ns = lmesh.findNeighbors({iOct,false}, neigh);
        neighbors_view(iOct, 0) = ns.size();
        for(int k=0; k<ns.size(); k++)
        {
          neighbors_view(iOct, k+1) = ns[k].iOct;
        }
      }
    }); 

    auto neighbors_view_host = Kokkos::create_mirror_view(neighbors_view);
    Kokkos::fence();

    time_lmesh_findneighbors.stop();
    std::cout << "Done in " << time_lmesh_findneighbors.elapsed()*1000 << " ms (cpu after fence)" << std::endl;
  }

}

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_CASE(test_LightOctree_2D)
{

  // always run this test
  run_test<2>();
  
} 

BOOST_AUTO_TEST_CASE(test_LightOctree_3D)
{

  // allow this test to be manually disabled
  // if there is an addition argument, disable
  if (boost::unit_test::framework::master_test_suite().argc==1)
    run_test<3>();
  
} 


BOOST_AUTO_TEST_CASE(test_LightOctree_hashmap_perf) 
// This test is disabled by default, enable it with 
{
  test_perf();  
}



BOOST_AUTO_TEST_SUITE_END() /* dyablo */

