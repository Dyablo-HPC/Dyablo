/**
 * \file test_LoadBalance.cpp
 * \author A. Durocher
 * Tests loadbalance MPI communication
 */


#include <boost/test/unit_test.hpp>

#include "legacy/utils_block.h"
#include "amr/AMRmesh.h"
#include "mpi/GhostCommunicator.h"

namespace dyablo
{

namespace muscl_block
{

// =======================================================================
// =======================================================================
void run_test(int argc, char *argv[])
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing LoadBalance ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    int ndim = 3;
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, 3, 5);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);

    amr_mesh->adaptGlobalRefine();
    amr_mesh->adaptGlobalRefine();
    amr_mesh->adaptGlobalRefine();

    uint8_t levels = 4;
    amr_mesh->adapt();
    amr_mesh->loadBalance(levels);
    amr_mesh->updateConnectivity();

    if( amr_mesh->getRank() == 0 )
    {
      // Refine initial 47 (final smaller 47..54)
      amr_mesh->setMarker(47,1);
      // Refine around initial 78
      amr_mesh->setMarker(65 ,1);
      amr_mesh->setMarker(72 ,1);
      amr_mesh->setMarker(73 ,1);
      amr_mesh->setMarker(67 ,1);
      amr_mesh->setMarker(74 ,1);
      amr_mesh->setMarker(75 ,1);
      amr_mesh->setMarker(76 ,1);
      amr_mesh->setMarker(77 ,1);
      amr_mesh->setMarker(71 ,1);
      amr_mesh->setMarker(69 ,1);
      //78
      amr_mesh->setMarker(81 ,1);
      amr_mesh->setMarker(88 ,1);
      amr_mesh->setMarker(89 ,1);
      amr_mesh->setMarker(79 ,1);
      amr_mesh->setMarker(85 ,1);
      amr_mesh->setMarker(92 ,1);
      amr_mesh->setMarker(93 ,1);
      amr_mesh->setMarker(97 ,1);
      amr_mesh->setMarker(104 ,1);
      amr_mesh->setMarker(105 ,1);
      amr_mesh->setMarker(99 ,1);
      amr_mesh->setMarker(106 ,1);
      amr_mesh->setMarker(107 ,1);
      amr_mesh->setMarker(113 ,1);
      amr_mesh->setMarker(120 ,1);
      amr_mesh->setMarker(121 ,1);
    }

    amr_mesh->adapt();
    amr_mesh->updateConnectivity();
  }

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  uint32_t nbCellsPerOct = bx*by*bz;
  uint32_t nbfields = 3;
  uint32_t nbOcts = amr_mesh->getNumOctants();
  uint32_t nbGhosts = amr_mesh->getNumGhosts();

  std::cout << "Initialize User Data..." << std::endl;
  DataArrayBlock U("U", nbCellsPerOct, nbfields, nbOcts );
  { // Initialize U
    DataArrayBlock::HostMirror U_host = Kokkos::create_mirror_view(U);
    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      bitpit::darray3 oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct);
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        U_host(c, IX, iOct) = oct_pos[IX] + cx*oct_size/bx;
        U_host(c, IY, iOct) = oct_pos[IY] + cy*oct_size/by;
        U_host(c, IZ, iOct) = oct_pos[IZ] + cz*oct_size/bz;
      }
    }
    Kokkos::deep_copy( U, U_host );
  }
  DataArrayBlock Ughost("Ughost", nbCellsPerOct, nbfields, nbGhosts );
  { // Initialize Ughost
    DataArrayBlock::HostMirror Ughost_host = Kokkos::create_mirror_view(Ughost);
    for( uint32_t iOct=0; iOct<nbGhosts; iOct++ )
    {
      bitpit::darray3 oct_pos = amr_mesh->getCoordinatesGhost(iOct);
      real_t oct_size = amr_mesh->getSizeGhost(iOct);
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        Ughost_host(c, IX, iOct) = oct_pos[IX] + cx*oct_size/bx;
        Ughost_host(c, IY, iOct) = oct_pos[IY] + cy*oct_size/by;
        Ughost_host(c, IZ, iOct) = oct_pos[IZ] + cz*oct_size/bz;
      }
    }
    Kokkos::deep_copy( Ughost, Ughost_host );
  }

  std::cout << "Perform load balancing..." << std::endl;
  {
    uint8_t levels = 4;

    amr_mesh->loadBalance_userdata(levels, U);
  }

  // Test U
  {
    uint32_t nbOcts = amr_mesh->getNumOctants();

    std::cout << "Check U ( nbOcts=" << nbOcts << ")" << std::endl;

    BOOST_CHECK_EQUAL(U.extent(0), nbCellsPerOct);
    BOOST_CHECK_EQUAL(U.extent(1), nbfields);
    BOOST_CHECK_EQUAL(U.extent(2), nbOcts);

    auto Uhost = Kokkos::create_mirror_view(U);
    Kokkos::deep_copy(Uhost, U);

    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      bitpit::darray3 oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct);
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        real_t expected_x = oct_pos[IX] + cx*oct_size/bx;
        real_t expected_y = oct_pos[IY] + cy*oct_size/by;
        real_t expected_z = oct_pos[IZ] + cz*oct_size/bz;

        BOOST_CHECK_CLOSE( Uhost(c, IX, iOct), expected_x , 0.01);
        BOOST_CHECK_CLOSE( Uhost(c, IY, iOct), expected_y , 0.01);
        BOOST_CHECK_CLOSE( Uhost(c, IZ, iOct), expected_z , 0.01);
      }
    }
  }

} // run_test

} // namespace muscl_block

} // namespace dyablo

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_SUITE(muscl_block)

BOOST_AUTO_TEST_CASE(test_GhostCommunicator)
{

  run_test(boost::unit_test::framework::master_test_suite().argc,
           boost::unit_test::framework::master_test_suite().argv);

}

BOOST_AUTO_TEST_SUITE_END() /* muscl_block */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */