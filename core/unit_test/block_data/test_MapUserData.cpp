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


#include "gtest/gtest.h"

#include "amr/MapUserData.h"

#include "amr/AMRmesh.h"
#include "amr/LightOctree.h"
#include "io/IOManager.h"

namespace dyablo
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

  int level_min = 1;
  int level_max = 8;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, level_min, level_max );
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

  Timers timers;

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
    "by=8\n";
  char* config_str_ptr = config_str;
  ConfigMap configMap(config_str_ptr, strlen(config_str)); //Use default values

  configMap.getValue<int>("mesh", "ndim", ndim);
  configMap.getValue<int>("amr", "bz", ndim==2?1:8);
  
  ForeachCell foreach_cell( *amr_mesh, configMap );

  std::string mapUserData_id = configMap.getValue<std::string>("amr", "remap", "MapUserData_legacy");
  std::unique_ptr<MapUserData> mapUserData = MapUserDataFactory::make_instance( mapUserData_id,
    configMap,
    foreach_cell,
    timers
  );
  
  uint32_t nbCellsPerOct = bx*by*bz;
  FieldManager field_manager({IU,IV,IW});
  uint32_t nbfields = field_manager.nbfields();
  uint32_t nbOcts = amr_mesh->getNumOctants();

  ForeachCell::CellArray_global_ghosted U = foreach_cell.allocate_ghosted_array("U", field_manager);

  { // Initialize U
    DataArrayBlock::HostMirror U_host = Kokkos::create_mirror_view(U.U);
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
    Kokkos::deep_copy( U.U, U_host );
  }

  // Ughost must be initialized when using MPI because coarsened blocks can use ghost values
  // Instead of MPI communication, Ughost is directly filled with cell positions
  { // Initialize U
    DataArrayBlock::HostMirror Ughost_host = Kokkos::create_mirror_view(U.Ughost);
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
    Kokkos::deep_copy( U.Ughost, Ughost_host );
  }

  std::string iomanager_id = "IOManager_hdf5";
  std::unique_ptr<IOManager> io_manager = IOManagerFactory::make_instance( iomanager_id,
    configMap,
    foreach_cell,
    timers
  );
  io_manager->save_snapshot(U, 0, 1);

  mapUserData->save_old_mesh();
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
  
  ForeachCell::CellArray_global_ghosted Unew = foreach_cell.allocate_ghosted_array("Unew", field_manager);

  std::cout << "Remap user data..." << std::endl;

  mapUserData->remap(U, Unew);

  {
    io_manager->save_snapshot(Unew, 1, 2);

    uint32_t nbOcts = amr_mesh->getNumOctants();

    std::cout << "Check Unew ( nbOcts=" << nbOcts << ")" << std::endl;

    EXPECT_EQ(Unew.U.extent(0), nbCellsPerOct);
    EXPECT_EQ(Unew.U.extent(1), nbfields);
    EXPECT_EQ(Unew.U.extent(2), nbOcts);

    auto Unew_host = Kokkos::create_mirror_view(Unew.U);
    Kokkos::deep_copy(Unew_host, Unew.U);

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


        EXPECT_NEAR( Unew_host(c, IX, iOct), expected_x , 0.0001);
        EXPECT_NEAR( Unew_host(c, IY, iOct), expected_y , 0.0001);
        EXPECT_NEAR( Unew_host(c, IZ, iOct), expected_z , 0.0001);
      }
    }
  }

} // run_test



} // namespace dyablo

TEST(dyablo, test_MapUserData_2D)
{

  dyablo::run_test(2);

}

TEST(dyablo, test_MapUserData_3D)
{

  dyablo::run_test(3);

}