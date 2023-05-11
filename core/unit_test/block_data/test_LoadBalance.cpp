/**
 * \file test_LoadBalance.cpp
 * \author A. Durocher
 * Tests loadbalance MPI communication
 */

#include "gtest/gtest.h"

#include "legacy/utils_block.h"
#include "amr/AMRmesh.h"
#include "mpi/GhostCommunicator.h"

#include "foreach_cell/ForeachCell.h"
#include "UserData.h"

namespace dyablo
{

// =======================================================================
// =======================================================================
void run_test()
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

  ConfigMap configMap(R"ini(
[mesh]
ndim=3
[amr]
bx=8
by=8
bz=8
level_min=3
level_max=5
)ini");

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  uint32_t nbCellsPerOct = bx*by*bz;
  uint32_t nbfields = 3;

  ForeachCell foreach_cell( *amr_mesh, configMap );
  UserData U( configMap, foreach_cell );
  U.new_fields({"px", "py", "pz"});

  std::cout << "Initialize User Data..." << std::endl;
  { // Initialize U
    enum VarIndex_test{Px,Py,Pz};

    UserData::FieldAccessor Uin = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}} );
    const ForeachCell::CellMetaData& cells = foreach_cell.getCellMetaData();
    foreach_cell.foreach_cell( "Init_U", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto c = cells.getCellCenter( iCell );
      Uin.at(iCell, Px) = c[IX];
      Uin.at(iCell, Py) = c[IY];
      Uin.at(iCell, Pz) = c[IZ];
    });
    U.exchange_ghosts( GhostCommunicator_kokkos( amr_mesh ) );
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

    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      auto oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct)[0];
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        real_t expected_x = oct_pos[IX] + cx*oct_size/bx;
        real_t expected_y = oct_pos[IY] + cy*oct_size/by;
        real_t expected_z = oct_pos[IZ] + cz*oct_size/bz;

        EXPECT_NEAR( Uhost_px(c, 0, iOct), expected_x , 0.01);
        EXPECT_NEAR( Uhost_py(c, 0, iOct), expected_y , 0.01);
        EXPECT_NEAR( Uhost_pz(c, 0, iOct), expected_z , 0.01);
      }
    }
  }

} // run_test



} // namespace dyablo


TEST(dyablo, test_GhostCommunicator)
{
  dyablo::run_test();
}