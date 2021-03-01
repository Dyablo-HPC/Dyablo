/**
 * \file test_CommGhosts.cpp
 * \author A. Durocher
 * Tests ghost octants MPI communication
 */


#include <boost/test/unit_test.hpp>

#include "shared/mpi/GhostCommunicator.h"

#include "muscl_block/utils_block.h"
#include "shared/amr/AMRmesh.h"
#include "utils/io/AMRMesh_output_vtk.h"

namespace dyablo
{

namespace muscl_block
{

// =======================================================================
// =======================================================================
void run_test(int argc, char *argv[])
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing GhostCommunicator ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    int ndim = 3;
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, 3, 7);
    //amr_mesh->setBalanceCodimension(ndim);
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
    amr_mesh->loadBalance();

    debug::output_vtk("before_initial", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(239 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt1", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(244 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt2", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(256 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt3", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(268 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt4", *amr_mesh);

    //amr_mesh->loadBalance();
    amr_mesh->updateConnectivity();
  }

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  uint32_t nbCellsPerOct = bx*by*bz;
  uint32_t nbfields = 3;
  uint32_t nbOcts = amr_mesh->getNumOctants();

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

  dyablo::muscl_block::GhostCommunicator ghost_communicator( amr_mesh );

  DataArrayBlock Ughost; //solver->Ughost
  ghost_communicator.exchange_ghosts(U, Ughost);

  // Test Ughost
  {
    uint32_t nGhosts = amr_mesh->getNumGhosts();

    std::cout << "Check Ughost ( nGhosts=" << nGhosts << ")" << std::endl;

    BOOST_CHECK_EQUAL(Ughost.extent(0), nbCellsPerOct);
    BOOST_CHECK_EQUAL(Ughost.extent(1), nbfields);
    BOOST_CHECK_EQUAL(Ughost.extent(2), nGhosts);

    auto Ughost_host = Kokkos::create_mirror_view(Ughost);
    Kokkos::deep_copy(Ughost_host, Ughost);

    for( uint32_t iGhost=0; iGhost<nGhosts; iGhost++ )
    {
      bitpit::darray3 oct_pos = amr_mesh->getCoordinatesGhost(iGhost);
      real_t oct_size = amr_mesh->getSizeGhost(iGhost);
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        real_t expected_x = oct_pos[IX] + cx*oct_size/bx;
        real_t expected_y = oct_pos[IY] + cy*oct_size/by;
        real_t expected_z = oct_pos[IZ] + cz*oct_size/bz;

        BOOST_CHECK_CLOSE( Ughost_host(c, IX, iGhost), expected_x , 0.01);
        BOOST_CHECK_CLOSE( Ughost_host(c, IY, iGhost), expected_y , 0.01);
        BOOST_CHECK_CLOSE( Ughost_host(c, IZ, iGhost), expected_z , 0.01);
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