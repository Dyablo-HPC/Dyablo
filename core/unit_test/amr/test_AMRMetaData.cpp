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

#include <iostream>

using Device = Kokkos::DefaultExecutionSpace;

#include <boost/test/unit_test.hpp>
using namespace boost::unit_test;

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

  // ======================================
  // stage 2 : create a AMRMetaData object
  // ======================================

  uint64_t capacity = 1024*1024;
  AMRMetaData<dim> amrMetadata(capacity);

  amrMetadata.report();
  amrMetadata.update_hashmap(amr_mesh);
  //amrMetadata.update_neigh_level_status(amr_mesh);
  amrMetadata.report();

  {

    // device map
    auto map_device = amrMetadata.hashmap();

    // host mirror
    using hashmap_host_t = typename AMRMetaData<dim>::hashmap_t::HostMirror;
    hashmap_host_t map_host(map_device.capacity());

    // copy on host before printing
    Kokkos::deep_copy(map_host, map_device);

    std::cout << "// ===========================================\n";
    std::cout << "// Print hashmap\n";
    std::cout << "// ===========================================\n";
    for (std::size_t i=0; i<map_host.capacity(); ++i)
    {
      if (map_host.valid_at(i)) 
      {
        auto key   = map_host.key_at(i)[0];
        auto level = map_host.key_at(i)[1];
        auto value = map_host.value_at(i);

        std::cout << std::setw(8) << i << " "
                  << "map[" << std::setw(18) << key << "]=" << std::setw(3) << value
                  << " (level=" << level << ")"
                  << " and Morton (from Pablo) = " << amr_mesh.getMorton(value)
                  << "\n";
      }
    }

    std::cout << "// ===========================================\n";
    std::cout << "// Print hashmap again (extract bits modulo 3)\n";
    std::cout << "// ===========================================\n";
    for (std::size_t i=0; i<map_host.capacity(); ++i)
    {
      if (map_host.valid_at(i)) 
      {
        auto key   = map_host.key_at(i)[0];
        auto level = map_host.key_at(i)[1];
        auto value = map_host.value_at(i);
        
        std::cout << std::setw(8) << i << " "
                  << "map[" << std::setw(6) << morton_extract_bits<3,IX>(key) 
                  << ","    << std::setw(6) << morton_extract_bits<3,IY>(key)
                  << ","    << std::setw(6) << morton_extract_bits<3,IZ>(key)
                  << "]=" << std::setw(3) << value
                  << " (level=" << level << ")"
                  << " and Morton (from Pablo) = " << amr_mesh.getMorton(value)
                  << "\n";
      }
    }

    std::cout << "// =========================================\n";
    std::cout << "// Print octant Ids and level\n";
    std::cout << "// =========================================\n";
    
    auto levels = amrMetadata.levels();
    typename AMRMetaData<dim>::levels_array_t::HostMirror levels_host = 
      Kokkos::create_mirror_view(levels);
    Kokkos::deep_copy(levels_host, levels);
    

    for (std::size_t iOct=0; iOct<amr_mesh.getNumOctants(); ++iOct)
    {
      std::cout << "iOct = " << iOct 
                << " level=" << int(levels_host(iOct))
                << "\n";
    }

  }

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

  {
    std::cout << "update neighbor status\n";
    amrMetadata.update_neigh_level_status(amr_mesh);

    // get neighbor level information
    const auto neigh_level_status = amrMetadata.neigh_level_status_array();

    typename AMRMetaData<dim>::neigh_level_status_array_t::HostMirror neigh_level_status_host = 
      Kokkos::create_mirror_view(neigh_level_status);

    Kokkos::deep_copy(neigh_level_status_host,
                      neigh_level_status);

    // get neighbor relative position information
    const auto neigh_rel_pos_status = amrMetadata.neigh_rel_pos_status_array();

    typename AMRMetaData<dim>::neigh_rel_pos_status_array_t::HostMirror neigh_rel_pos_status_host = 
      Kokkos::create_mirror_view(neigh_rel_pos_status);

    Kokkos::deep_copy(neigh_rel_pos_status_host,
                      neigh_rel_pos_status);

    // print neighbor information
    for (std::size_t iOct=0; iOct<amr_mesh.getNumOctants(); ++iOct)
    {
      auto status = neigh_level_status_host(iOct);
      std::bitset<8*sizeof(status)> status_binary(status);

      auto status2 = neigh_rel_pos_status_host(iOct);
      std::bitset<8*sizeof(status2)> status2_binary(status2);
      
      std::cout << "iOct " << iOct << " | " 
                << "status=" << status 
                << " ( " << status_binary << " ) | "
                << "status2=" << int(status2) 
                << " ( " << status2_binary << " )\n" ;

      amrMetadata.decode_neighbor_status(iOct,status,status2);
    }
  }

  // ============================================================
  // ============================================================

  std::cout << "// =========================================\n";
  std::cout << "// Print mesh face connectivity\n";
  std::cout << "// =========================================\n";

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

  {
    //
    // get hashmap
    //
    auto map_device = amrMetadata.hashmap();

    using hashmap_host_t = typename AMRMetaData<dim>::hashmap_t::HostMirror;
    
    auto invalid_index = ~static_cast<uint32_t>(0);

    hashmap_host_t map_host(map_device.capacity());

    Kokkos::deep_copy(map_host, map_device);

    //
    // get morton keys array
    //
    const auto morton_keys = amrMetadata.morton_keys();

    typename AMRMetaData<dim>::morton_keys_array_t::HostMirror morton_keys_host =
      Kokkos::create_mirror_view(morton_keys);
    
    Kokkos::deep_copy(morton_keys_host,
                      morton_keys);
    

    //
    // get neighbor level information
    //
    const auto neigh_level_status = amrMetadata.neigh_level_status_array();
    
    typename AMRMetaData<dim>::neigh_level_status_array_t::HostMirror neigh_level_status_host = 
      Kokkos::create_mirror_view(neigh_level_status);
    
    Kokkos::deep_copy(neigh_level_status_host,
                      neigh_level_status);
    
    const uint8_t nbFaces   = 2*dim;

    using NEIGH_LEVEL = typename AMRMetaData<dim>::NEIGH_LEVEL;

    // neighbor key
    typename AMRMetaData<dim>::key_t key_n; 

    for (std::size_t i=0; i<map_host.capacity(); ++i)
    {
      if (map_host.valid_at(i)) 
      {
        auto morton= map_host.key_at(i)[0];
        auto level = map_host.key_at(i)[1];
        auto iOct  = map_host.value_at(i);
        
        auto status = neigh_level_status_host(iOct);
        
        auto x = morton_extract_bits<3,IX>(morton);
        auto y = morton_extract_bits<3,IY>(morton);
        auto z = morton_extract_bits<3,IZ>(morton);

        std::cout << "iOct " << std::setw(3) << iOct << " (morton=" << std::setw(19) << morton << ") ";
        std::cout << "| " << std::setw(7) << x 
                  << " "  << std::setw(7) << y 
                  << " "  << std::setw(7) << z << " |";
        std::cout << " || "; 
        
        for (int iface=0; iface<nbFaces; ++iface)
        {
          NEIGH_LEVEL nl = static_cast<NEIGH_LEVEL>( (status >> (2*iface)) & 0x3 );
          
          if (nl == NEIGH_LEVEL::NEIGH_IS_SAME_SIZE)
          {
            key_n[0] = get_neighbor_morton(morton,level,iface);
            key_n[1] = level;

            auto index_n = map_host.find(key_n);

            if ( invalid_index != index_n )
            {
              // print neighbor iOct
              std::cout << "| f" << iface << " " << map_host.value_at(index_n) << " ";
            }
            else
            {
              std::cout << "Invalid index when trying to access hashmap...\n";
            }

          }

          if (nl == NEIGH_LEVEL::NEIGH_IS_SMALLER)
          {
            // max number of neighbors through a face
            const uint8_t nbNeighs = 1<<(dim-1);

            std::cout << "| f" << iface << " " ;

            for (uint8_t ineigh=0; ineigh<nbNeighs; ++ineigh)
            {

              auto level_n = level+1;
              key_n[0] = get_neighbor_morton(morton, level, level_n, iface, ineigh);
              key_n[1] = level_n;

              auto index_n = map_host.find(key_n);
              
              if ( invalid_index != index_n )
              {
                // print neighbor iOct
                std::cout << " " << map_host.value_at(index_n) << " ";
              }
              else
              {
                std::cout << "Invalid index when trying to access hashmap...\n";
              }
                            
            } // end for ineigh

          } // end if NEIGH_IS_SMALLER
          
          if (nl == NEIGH_LEVEL::NEIGH_IS_LARGER)
          {
            // max number of "possible location" neighbors through a face
            const uint8_t nbNeighs = 1<<(dim-1);

            std::cout << "| f" << iface << " " ;

            for (uint8_t ineigh=0; ineigh<nbNeighs; ++ineigh)
            {

              auto level_n = level-1;
              key_n[0] = get_neighbor_morton(morton, level, level_n, iface, ineigh);
              key_n[1] = level_n;

              auto index_n = map_host.find(key_n);
              
              if ( invalid_index != index_n )
              {
                // print neighbor iOct
                std::cout << " " << map_host.value_at(index_n) << " ";
              }
              // we don't need to print this
              // else
              // {
              //   std::cout << "Invalid index when trying to access hashmap...\n";
              // }
                            
            } // end for ineigh
            

          } // end if NEIGH_IS_LARGER
          
        } // end for iface

        std::cout << "\n";
      } // if map valid

    } // end map

  } // end print mesh connectivity

} // run_test

} // dyablo



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
  if (framework::master_test_suite().argc==1)
    run_test<3>();
  
} 

BOOST_AUTO_TEST_SUITE_END() /* dyablo */


// old main
#if 0
// ==========================================================
// ==========================================================
int main(int argc, char* argv[])
{
  
  // Create MPI session if MPI enabled
#ifdef DYABLO_USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc, &argv);
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
    if (Kokkos::hwloc::available()) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE[" << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT[" << Kokkos::hwloc::get_available_threads_per_core()
          << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";
    
#ifdef DYABLO_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#ifdef KOKKOS_ENABLE_CUDA
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
#endif // KOKKOS_ENABLE_CUDA
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

} // main

#endif // old main
