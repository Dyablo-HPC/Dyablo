/**
 * \file test_MapUserData.cpp
 * \author A. Durocher
 * Test user data remapping after amr cycle
 * 
 * Create an AMR mesh and fill user data with cell positions
 * Refine/coarsen mesh and then test if new user data contains the right positions
 * 
 * This also works with multiple MPI process
 */


#include <boost/test/unit_test.hpp>

#include "muscl_block/MapUserData.h"

#include "muscl_block/utils_block.h"
#include "shared/amr/AMRmesh.h"
#include "shared/amr/LightOctree.h"
#include "muscl_block/io/IOManager.h"

namespace dyablo
{

namespace muscl_block
{

// =======================================================================
// =======================================================================
void run_test(int ndim)
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing MapUserData ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = (ndim==3)?8:1;

  HydroParams params;
  params.dimType = (ndim == 2) ? TWO_D : THREE_D;
  params.level_min = 1;
  params.level_max = 8;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, params.level_min, params.level_max );
    // amr_mesh->setBalanceCodimension(ndim);
    // uint32_t idx = 0;
    // amr_mesh->setBalance(idx,true);
    // amr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    // amr_mesh->setPeriodic(4);
    // amr_mesh->setPeriodic(5);

    amr_mesh->adaptGlobalRefine();
    amr_mesh->adaptGlobalRefine();
    amr_mesh->adaptGlobalRefine();

    amr_mesh->adapt();
    amr_mesh->updateConnectivity();
    amr_mesh->loadBalance();
    amr_mesh->updateConnectivity();
  }

  
  uint32_t nbCellsPerOct = bx*by*bz;
  FieldManager field_manager({IU,IV,IW});
  uint32_t nbfields = field_manager.nbfields();
  auto fm = field_manager.get_id2index();
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

        U_host(c, IX, iOct) = oct_pos[IX] + (cx+0.5)*oct_size/bx;
        U_host(c, IY, iOct) = oct_pos[IY] + (cy+0.5)*oct_size/by;
        U_host(c, IZ, iOct) = oct_pos[IZ] + (cz+0.5)*oct_size/bz;
      }
    }
    Kokkos::deep_copy( U, U_host );
  }

  // Ughost must be initialized when using MPI because coarsened blocks can use ghost values
  // Instead of MPI communication, Ughost is directly filled with cell positions
  DataArrayBlock Ughost("Ughost", nbCellsPerOct, nbfields, amr_mesh->getNumGhosts() );
  { // Initialize U
    DataArrayBlock::HostMirror Ughost_host = Kokkos::create_mirror_view(Ughost);
    for( uint32_t iOct=0; iOct<amr_mesh->getNumGhosts(); iOct++ )
    {
      bitpit::darray3 oct_pos = amr_mesh->getCoordinatesGhost(iOct);
      real_t oct_size = amr_mesh->getSizeGhost(iOct);
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        Ughost_host(c, IX, iOct) = oct_pos[IX] + (cx+0.5)*oct_size/bx;
        Ughost_host(c, IY, iOct) = oct_pos[IY] + (cy+0.5)*oct_size/by;
        Ughost_host(c, IZ, iOct) = oct_pos[IZ] + (cz+0.5)*oct_size/bz;
      }
    }
    Kokkos::deep_copy( Ughost, Ughost_host );
  }

  char config_str[] = 
    "[output]\n"
    "hdf5_enabled=true\n"
    "write_mesh_info=true\n"
    "write_variables=rho_vx,rho_vy,rho_vz\n"
    "write_iOct=false\n"
    "outputPrefix=output\n"
    "outputDir=./\n"
    "[amr]\n"
    "use_block_data=true\n"
    "bx=8\n"
    "by=8\n"
    "bz=8\n";
  char* config_str_ptr = config_str;
  ConfigMap configMap(config_str_ptr, strlen(config_str)); //Use default values

  Timers timers;
  std::string iomanager_id = "IOManager_hdf5";
  std::unique_ptr<IOManager> io_manager = IOManagerFactory::make_instance( iomanager_id,
    configMap,
    params,
    *amr_mesh, 
    field_manager,
    bx, by, bz,
    timers
  );
  io_manager->save_snapshot(U, Ughost, 0, 1);

  LightOctree lmesh_old = amr_mesh->getLightOctree();
  {
    std::cout << "Coarsen/Refine octants" << std::endl;

     for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
     {
       amr_mesh->setMarker(iOct , -1);
       //amr_mesh->setMarker(iOct , 1); // replate previous line with this to check refinement thorougly
     }

    // // Refine 0 because it is at an MPI boundary 
    // // 2:1 balance should create cells that are neither coarsened nor refined
    amr_mesh->setMarker((uint32_t)0 , 1);
    // // Refine a random octant in the middle
    amr_mesh->setMarker(nbOcts/2 , 1);

    amr_mesh->adapt(true);
  }
  const LightOctree& lmesh_new = amr_mesh->getLightOctree();

  
  DataArrayBlock Unew;

  std::cout << "Remap user data..." << std::endl;

  MapUserDataFunctor::apply(lmesh_old, lmesh_new,
                            {bx, by, bz}, 
                            U, Ughost, Unew);

  {
    io_manager->save_snapshot(Unew, Ughost, 1, 2);

    uint32_t nbOcts = amr_mesh->getNumOctants();

    std::cout << "Check Unew ( nbOcts=" << nbOcts << ")" << std::endl;

    BOOST_CHECK_EQUAL(Unew.extent(0), nbCellsPerOct);
    BOOST_CHECK_EQUAL(Unew.extent(1), nbfields);
    BOOST_CHECK_EQUAL(Unew.extent(2), nbOcts);

    auto Unew_host = Kokkos::create_mirror_view(Unew);
    Kokkos::deep_copy(Unew_host, Unew);

    real_t oct_size_initial = 1./8; 

    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      bitpit::darray3 oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct);
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        real_t expected_x = oct_pos[IX] + (cx+0.5)*oct_size/bx;
        real_t expected_y = oct_pos[IY] + (cy+0.5)*oct_size/by;
        real_t expected_z = oct_pos[IZ] + (cz+0.5)*oct_size/bz; 

        if(oct_size < oct_size_initial)
        { // When cell is newly refined there is no interpolation
          expected_x += ((cx%2)?-1:1) * oct_size/(2*bx);
          expected_y += ((cy%2)?-1:1) * oct_size/(2*by);
          expected_z += ((cz%2)?-1:1) * oct_size/(2*bz);
        }

        if(ndim==2)
          expected_z = oct_size_initial/(2*bz);


        BOOST_CHECK_CLOSE( Unew_host(c, IX, iOct), expected_x , 0.0001);
        BOOST_CHECK_CLOSE( Unew_host(c, IY, iOct), expected_y , 0.0001);
        BOOST_CHECK_CLOSE( Unew_host(c, IZ, iOct), expected_z , 0.0001);
      }
    }
  }

} // run_test

} // namespace muscl_block

} // namespace dyablo

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_SUITE(muscl_block)

BOOST_AUTO_TEST_CASE(test_MapUserData_2D)
{

  run_test(2);

}

BOOST_AUTO_TEST_CASE(test_MapUserData_3D)
{

  run_test(3);

}

BOOST_AUTO_TEST_SUITE_END() /* muscl_block */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */