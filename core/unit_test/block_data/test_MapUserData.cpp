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
void run_test(int ndim, std::string mapUserData_id)
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing MapUserData ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = (ndim==3)?8:1;

  int level_min = 2;
  int level_max = 8;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, level_min, level_max );
    amr_mesh->adaptGlobalRefine();
    // amr_mesh->setBalanceCodimension(ndim);
    // uint32_t idx = 0;
    // amr_mesh->setBalance(idx,true);
    // amr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    // amr_mesh->setPeriodic(4);
    // amr_mesh->setPeriodic(5);
  }

  Timers timers;

  std::string config_str = 
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
  ConfigMap configMap(config_str); //Use default values

  configMap.getValue<int>("mesh", "ndim", ndim);
  configMap.getValue<int>("amr", "bz", ndim==2?1:8);
  
  ForeachCell foreach_cell( *amr_mesh, configMap );

  std::unique_ptr<MapUserData> mapUserData = MapUserDataFactory::make_instance( mapUserData_id,
    configMap,
    foreach_cell,
    timers
  );
  
  uint32_t nbCellsPerOct = bx*by*bz;
  uint32_t nbOcts = amr_mesh->getNumOctants();

  UserData U( configMap, foreach_cell );
  U.new_fields({"px", "py", "pz"});
  uint32_t nbfields = U.nbFields();

  std::cout << "Initialize User Data..." << std::endl;
  { // Initialize U
    DataArrayBlock::HostMirror U_host_px = Kokkos::create_mirror_view(U.getField("px").U);
    DataArrayBlock::HostMirror U_host_py = Kokkos::create_mirror_view(U.getField("py").U);
    DataArrayBlock::HostMirror U_host_pz = Kokkos::create_mirror_view(U.getField("pz").U);
    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      auto oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct)[0];
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        U_host_px(c, 0, iOct) = oct_pos[IX] + (cx+0.5)*oct_size/bx;
        U_host_py(c, 0, iOct) = oct_pos[IY] + (cy+0.5)*oct_size/by;
        U_host_pz(c, 0, iOct) = oct_pos[IZ] + (cz+0.5)*oct_size/bz;
      }
    }
    Kokkos::deep_copy( U.getField("px").U, U_host_px );
    Kokkos::deep_copy( U.getField("py").U, U_host_py );
    Kokkos::deep_copy( U.getField("pz").U, U_host_pz );
  }

  // Ughost must be initialized when using MPI because coarsened blocks can use ghost values
  // Instead of MPI communication, Ughost is directly filled with cell positions
  { // Initialize U
    DataArrayBlock::HostMirror U_ghost_host_px = Kokkos::create_mirror_view(U.getField("px").Ughost);
    DataArrayBlock::HostMirror U_ghost_host_py = Kokkos::create_mirror_view(U.getField("py").Ughost);
    DataArrayBlock::HostMirror U_ghost_host_pz = Kokkos::create_mirror_view(U.getField("pz").Ughost);
    for( uint32_t iOct=0; iOct<amr_mesh->getNumGhosts(); iOct++ )
    {
      auto oct_pos = amr_mesh->getCoordinatesGhost(iOct);
      real_t oct_size = amr_mesh->getSizeGhost(iOct)[0];
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        U_ghost_host_px(c, 0, iOct) = oct_pos[IX] + (cx+0.5)*oct_size/bx;
        U_ghost_host_py(c, 0, iOct) = oct_pos[IY] + (cy+0.5)*oct_size/by;
        U_ghost_host_pz(c, 0, iOct) = oct_pos[IZ] + (cz+0.5)*oct_size/bz;
      }
    }
    Kokkos::deep_copy( U.getField("px").Ughost, U_ghost_host_px );
    Kokkos::deep_copy( U.getField("py").Ughost, U_ghost_host_py );
    Kokkos::deep_copy( U.getField("pz").Ughost, U_ghost_host_pz );
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
  
  std::cout << "Remap user data..." << std::endl;

  mapUserData->remap(U);

  {
    io_manager->save_snapshot(U, 1, 2);

    uint32_t nbOcts = amr_mesh->getNumOctants();

    std::cout << "Check remapped U ( nbOcts=" << nbOcts << ")" << std::endl;

    EXPECT_EQ( U.nbFields(), nbfields );

    EXPECT_EQ( U.getField("px").nbOcts, nbOcts );
    EXPECT_EQ( U.getField("py").nbOcts, nbOcts );
    EXPECT_EQ( U.getField("pz").nbOcts, nbOcts );

    uint32_t expected_size = nbOcts*bx*by*bz;
    EXPECT_EQ( U.getField("px").U.size(), expected_size );
    EXPECT_EQ( U.getField("py").U.size(), expected_size );
    EXPECT_EQ( U.getField("pz").U.size(), expected_size );

    auto Uhost_px = Kokkos::create_mirror_view(U.getField("px").U);
    auto Uhost_py = Kokkos::create_mirror_view(U.getField("py").U);
    auto Uhost_pz = Kokkos::create_mirror_view(U.getField("pz").U);
    Kokkos::deep_copy(Uhost_px, U.getField("px").U);
    Kokkos::deep_copy(Uhost_py, U.getField("py").U);
    Kokkos::deep_copy(Uhost_pz, U.getField("pz").U);

    real_t oct_size_initial = 1./8; 

    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      auto oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct)[0];
      
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


        EXPECT_NEAR( Uhost_px(c, 0, iOct), expected_x , 0.0001);
        EXPECT_NEAR( Uhost_py(c, 0, iOct), expected_y , 0.0001);
        EXPECT_NEAR( Uhost_pz(c, 0, iOct), expected_z , 0.0001);
      }
    }
  }

} // run_test

} // namespace dyablo

class Test_MapUserData
  : public testing::TestWithParam<std::tuple<int, std::string>> 
{};

TEST_P(Test_MapUserData, position_field_conserved)
{
  int ndim = std::get<0>(GetParam());
  std::string id = std::get<1>(GetParam());
  dyablo::run_test(ndim, id );
}

INSTANTIATE_TEST_SUITE_P(
    Test_MapUserData, Test_MapUserData,
    testing::Combine(
        testing::Values(2,3),
        testing::ValuesIn( dyablo::MapUserDataFactory::get_available_ids() )
    ),
    [](const testing::TestParamInfo<Test_MapUserData::ParamType>& info) {
      std::string name = 
          (std::get<0>(info.param) == 2 ? std::string("2D") : std::string("3D"))
          + "_" + std::get<1>(info.param);
      return name;
    }
);
