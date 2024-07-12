/**
 * \file test_CommGhosts.cpp
 * \author A. Durocher
 * Tests ghost octants MPI communication
 */
#include "gtest/gtest.h"

#include "mpi/ViewCommunicator.h"

#include "legacy/utils_block.h"
#include "amr/AMRmesh.h"
#include "utils/io/AMRMesh_output_vtk.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

#include "mpi/GhostCommunicator.h"
#include "UserData.h"
#include "utils/config/ConfigMap.h"

void test_GhostCommunicator_partial_block()
{
  using namespace dyablo;

  std::cout << "// =========================================\n";
  std::cout << "// Testing GhostCommunicator_partial_block ...\n";
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

    debug::output_vtk("before_initial", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt1", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt2", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt3", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt4", *amr_mesh);
  }

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  ConfigMap configMap ("");
  configMap.getValue<uint32_t>("amr", "bx", bx);
  configMap.getValue<uint32_t>("amr", "by", by);
  configMap.getValue<uint32_t>("amr", "bz", bz);
  
  ForeachCell foreach_cell(*amr_mesh, configMap);  
  UserData U ( configMap, foreach_cell );

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
  }

  enum VarIndex_test{Px,Py,Pz};
  UserData::FieldAccessor Ua = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}} );

  GhostCommunicator ghost_communicator( *amr_mesh, U.getShape(), 2 );
  ghost_communicator.exchange_ghosts( Ua );

  // test that ghosts have the right value
  auto cells = foreach_cell.getCellMetaData();

  int error_count = 0;
  foreach_cell.reduce_cell( "test_neighbors", Ua.getShape(),
    KOKKOS_LAMBDA( ForeachCell::CellIndex& iCell, int& error_count )
  {
    auto check_value = [&](const CellIndex& iCell)
    {
      auto test_equal = [&](real_t a, real_t b)
      {
        //EXPECT_DOUBLE_EQ(a, b);
        if( a != b )
        {
          error_count++;
          printf("%f != %f", a, b);
        }
      };

      auto pos = cells.getCellCenter(iCell);
      test_equal( pos[IX], Ua.at(iCell, Px) );
      test_equal( pos[IY], Ua.at(iCell, Py) );
      test_equal( pos[IZ], Ua.at(iCell, Pz) );
    }; 

    auto test_offset = [&]( CellIndex::offset_t offset, int ghost_width )
    {
      CellIndex iCell_n = iCell.getNeighbor_ghost( offset, Ua );
      if( !iCell_n.is_local() && !iCell_n.is_boundary() )
      {
        if( iCell_n.level_diff() >= 0 )
        {
          check_value(iCell_n);

          // Check other neighbors (in same block)
          for( int i=1; i<ghost_width; i++ )
          {
            CellIndex::offset_t offset_nn{(int8_t)(offset[IX]*i),(int8_t)(offset[IY]*i),(int8_t)(offset[IZ]*i)};
            CellIndex iCell_nn = iCell_n.getNeighbor( offset_nn );

            check_value(iCell_nn);
          }
        }
        else
        {
          foreach_smaller_neighbor<3>( iCell_n, offset, Ua.getShape(),
          [&]( const CellIndex& iCell_ns )
          {
            check_value(iCell_ns);

            // Check other neighbors (in same block)
            for( int i=1; i<ghost_width; i++ )
            {
              CellIndex::offset_t offset_nn{(int8_t)(offset[IX]*i),(int8_t)(offset[IY]*i),(int8_t)(offset[IZ]*i)};
              CellIndex iCell_nn = iCell_ns.getNeighbor( offset_nn );

              check_value(iCell_nn);
            }
          });    
        }
      }
    };

    test_offset( { 1, 0, 0}, 2 );
    test_offset( {-1, 0, 0}, 2 );
    test_offset( { 0, 1, 0}, 2 );
    test_offset( { 0,-1, 0}, 2 );
    test_offset( { 0, 0, 1}, 2 );
    test_offset( { 0, 0,-1}, 2 );
    
  }, error_count);

  EXPECT_EQ(0, error_count);
}

TEST(dyablo, test_GhostCommunicator_partial_block)
{
  test_GhostCommunicator_partial_block();
}

void run_test_reduce_partial_blocks()
{
  using namespace dyablo;

  std::cout << "// =========================================\n";
  std::cout << "// Testing GhostCommunicator_partial_block reduce ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  constexpr int ndim = 3;
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{true,true,true}, 3, 7);
    //amr_mesh->setBalanceCodimension(ndim);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);

    debug::output_vtk("before_initial", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt1", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt2", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt3", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt4", *amr_mesh);

    amr_mesh->loadBalance();
  }

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  ConfigMap configMap ("");
  configMap.getValue<uint32_t>("amr", "bx", bx);
  configMap.getValue<uint32_t>("amr", "by", by);
  configMap.getValue<uint32_t>("amr", "bz", bz);
  
  ForeachCell foreach_cell(*amr_mesh, configMap);  
  UserData U ( configMap, foreach_cell );

  U.new_fields({"px", "py", "pz"});

  std::cout << "Initialize User Data..." << std::endl;

  {// Initialize U

    enum VarIndex_test{Px,Py,Pz};

    UserData::FieldAccessor Uin = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}} );

    foreach_cell.foreach_cell( "Fill_neighbors", U.getShape(),
      CELL_LAMBDA( const CellIndex& iCell )
    {
      auto fill_neighbor = [&](CellIndex::offset_t offset, VarIndex iVar)
      {
        CellIndex iCell_n = iCell.getNeighbor_ghost( offset, Uin );    
        assert( iCell_n.is_valid() ); // This test uses periodic
        assert( iCell_n.level_diff() >= -1 &&  iCell_n.level_diff() <= 1 );
        if( iCell_n.level_diff() == 0 ) // Same size
        {
          Kokkos::atomic_add(&Uin.at( iCell_n, iVar ), 1); 
        }
        else if( iCell_n.level_diff() == 1 ) // Neighbor is bigger
        {
          Kokkos::atomic_add(&Uin.at( iCell_n, iVar ), 0.25); 
        }
        else if( iCell_n.level_diff() == -1 ) // Neighbors are smaller
        {
          foreach_smaller_neighbor<ndim>( iCell_n, offset, Uin.getShape(),
            [&]( const CellIndex& iCell_ns )
          {
            Kokkos::atomic_add(&Uin.at( iCell_ns, iVar ), 1);
          });       
        }
        else
        {
          assert(false);
        }
      };
      fill_neighbor( { 1, 0, 0}, Px );
      fill_neighbor( {-1, 0, 0}, Px );
      fill_neighbor( { 0, 1, 0}, Py );
      fill_neighbor( { 0,-1, 0}, Py );
      fill_neighbor( { 0, 0, 1}, Pz );
      fill_neighbor( { 0, 0,-1}, Pz );
    });

  }

  enum VarIndex_test{Px,Py,Pz};
  UserData::FieldAccessor Ua = U.getAccessor( {{"px", Px}, {"py", Py}, {"pz", Pz}} );

  GhostCommunicator ghost_communicator( *amr_mesh, U.getShape(), 2 );
  ghost_communicator.reduce_ghosts( Ua );

  int nerrors = 0;
  foreach_cell.reduce_cell( "check_values", U.getShape(),
    CELL_LAMBDA( const CellIndex& iCell, int& nerrors )
  {
    constexpr real_t eps = 1e-10;
    real_t U_IU = Ua.at( iCell, Px );
    real_t U_IV = Ua.at( iCell, Py );
    real_t U_IW = Ua.at( iCell, Pz );

    #ifndef __CUDA_ARCH__
    EXPECT_NEAR( 2, U_IU, eps);
    EXPECT_NEAR( 2, U_IV, eps);
    EXPECT_NEAR( 2, U_IW, eps);
    #endif

    if( abs( U_IU - 2) > eps )
      nerrors++;
    if( abs( U_IV - 2) > eps )
      nerrors++;
    if( abs( U_IW - 2) > eps )
      nerrors++;
    
  }, nerrors);

  EXPECT_EQ(nerrors, 0);

} // run_test_reduce

TEST(dyablo, test_GhostCommunicator_reduce)
{
  run_test_reduce_partial_blocks();
}