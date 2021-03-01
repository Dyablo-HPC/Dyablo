/**
 * \file test_CommGhosts.cpp
 * \author A. Durocher
 * Tests ghost octants MPI communication
 */


#include <boost/test/unit_test.hpp>

#include "muscl_block/GhostCommunicator.h"

#include "muscl_block/utils_block.h"
#include "shared/amr/AMRmesh.h"

namespace dyablo
{

namespace muscl_block
{

template<typename T>
void print_array( std::ostream& out, const std::vector<T>& v )
{
  for( const auto& e : v )
    out << e << " ";
  out << std::endl;
}

void output_vtk( const AMRmesh& mesh )
{
  uint32_t nofCubes = mesh.getNumOctants() + mesh.getNumGhosts();

  std::vector<double> cell_data(nofCubes);

  std::vector<double> nodes_Coordinates(nofCubes*8*3);
  std::vector<uint32_t> cells_Connectivity(nofCubes*8);
  std::vector<uint32_t> cells_offsets(nofCubes);
  std::vector<int> cells_types(nofCubes, 11);

  for(uint32_t i=0; i<nofCubes; i++)
  {
    real_t px, py, pz, size;

    if( i<mesh.getNumOctants() )
    {
      px = mesh.getCoordinates(i)[IX];
      py = mesh.getCoordinates(i)[IY];
      pz = mesh.getCoordinates(i)[IZ];
      size = mesh.getSize(i);
    }
    else
    {
      uint32_t iOct = i-mesh.getNumOctants();
      px = mesh.getCoordinatesGhost(iOct)[IX];
      py = mesh.getCoordinatesGhost(iOct)[IY];
      pz = mesh.getCoordinatesGhost(iOct)[IZ];
      size = mesh.getSizeGhost(iOct);
      cell_data[i] = 1;
    }

    for( int16_t dz=0; dz<2; dz++ )
    for( int16_t dy=0; dy<2; dy++ )
    for( int16_t dx=0; dx<2; dx++ )
    {
      int di = dx + 2*dy + 4*dz;
      nodes_Coordinates[3*(8*i+di) + IX] = px + size * dx;
      nodes_Coordinates[3*(8*i+di) + IY] = py + size * dy;
      nodes_Coordinates[3*(8*i+di) + IZ] = pz + size * dz;
      cells_Connectivity[8*i+di] = 8*i+di;
    }

    cells_offsets[i] = 8*i+8;
  }

  std::string filename = std::string("mesh_test_GhostCommunicator_")+std::to_string(hydroSimu::GlobalMpiSession::getRank())+".vtu";

  std::ofstream out( filename );
  out << "<?xml version=\"1.0\"?>"                                                      << std::endl;
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
  out << "  <UnstructuredGrid>"                                                         << std::endl;
  out << "    <Piece NumberOfCells=\"" << nofCubes << "\" NumberOfPoints=\"" << nofCubes*8 << "\">" << std::endl;
  out << "      <Points>"                                                               << std::endl;
  out << "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  print_array(out, nodes_Coordinates);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </Points>"                                                              << std::endl;
  out << "      <CellData>"                                                             << std::endl;
  out << "        <DataArray type=\"Float64\" Name=\"data\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cell_data);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </CellData>"                                                            << std::endl;
  out << "      <Cells>"                                                                << std::endl;
  out << "        <DataArray type=\"UInt32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_Connectivity);
  out << "        </DataArray>"                                                         << std::endl;
  out << "        <DataArray type=\"UInt32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_offsets);
  out << "        </DataArray>"                                                         << std::endl;
  out << "        <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_types);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </Cells>"                                                               << std::endl;
  out << "    </Piece>"                                                                 << std::endl;
  out << "  </UnstructuredGrid>"                                                        << std::endl;
  out << "</VTKFile>"                                                                   << std::endl;

}

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
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, 3, 5);
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

    amr_mesh->adapt();
    amr_mesh->loadBalance();
    amr_mesh->updateConnectivity();

    output_vtk(*amr_mesh);
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