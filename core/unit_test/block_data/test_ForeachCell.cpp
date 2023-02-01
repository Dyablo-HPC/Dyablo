/**
 * \file test_LoadBalance.cpp
 * \author A. Durocher
 * Tests loadbalance MPI communication
 */

#include "gtest/gtest.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

#include "states/State_Nd.h"

namespace dyablo
{

// =======================================================================
// =======================================================================
void run_test()
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing ForeachCell ...\n";
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
      EXPECT_GT( amr_mesh->getNumOctants() , 121 ) << "Internal test error : not enough octants in MPI rank 0";

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
    amr_mesh->loadBalance();
  }

  // Content of .ini file used ton configure configmap and HydroParams
  std::string configmap_str = "";
  ConfigMap configMap(configmap_str);

  // Fill default values
  constexpr int ndim = 3;
  configMap.getValue<int>("run", "ndim", ndim);
  uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 8);
  uint32_t by = configMap.getValue<uint32_t>("amr", "by", 1);
  uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", (ndim==2)?1:4);
  configMap.getValue<real_t>("mesh", "xmin", -2);
  configMap.getValue<real_t>("mesh", "ymin", 0);
  configMap.getValue<real_t>("mesh", "zmin", 1);
  configMap.getValue<real_t>("mesh", "xmax", 2);
  configMap.getValue<real_t>("mesh", "ymax", 2);
  configMap.getValue<real_t>("mesh", "zmax", 4);

  // Create ForeachCell
  std::cout << "Initialize ForeachCell..." << std::endl;
  ForeachCell foreach_cell( *amr_mesh, configMap );
  ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

  std::cout << "Allocate U..." << std::endl;
  FieldManager field_manager({IU,IV,IW});
  ForeachCell::CellArray_global_ghosted U = foreach_cell.allocate_ghosted_array("U", field_manager);
  std::cout << "Fill U..." << std::endl;
  {
    foreach_cell.foreach_cell( "Fill_U", U, 
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto pos = cells.getCellCenter(iCell);
      U.at( iCell, IU ) = pos[IX];
      U.at( iCell, IV ) = pos[IY];
      U.at( iCell, IW ) = pos[IZ];
    });
  }

  std::cout << "Apply stencil..." << std::endl;
  ForeachCell::CellArray_global_ghosted U2 = foreach_cell.allocate_ghosted_array("U2", field_manager);
  foreach_cell.foreach_cell( "Stencil_U2", U, 
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
  {
    auto append_offset = [&]( ForeachCell::CellIndex::offset_t offset )
    {
      ForeachCell::CellIndex iCell_n = iCell.getNeighbor_ghost( offset, U );
      StateNd<3> offset_real = {(real_t)offset[IX], (real_t)offset[IY], (real_t)offset[IZ]};

      // Res will be center of same-size virtual neighbor
      StateNd<3> res = {}; // StateNd to use arithmetic operators
      if( iCell_n.is_boundary() )
      {
        // Compute neighbor position from local cell position
        auto pos = cells.getCellCenter(iCell);
        auto size = cells.getCellSize(iCell);
        res = pos + size*offset_real;
      }
      else
      {
        if( iCell_n.level_diff() == 0 )
        {
          // Just get same-size neighbor position
          auto pos = cells.getCellCenter(iCell_n);
          res = pos;
        }
        else if( iCell_n.level_diff() == -1 )
        {
          // Mean of positions, shifted to center of bigger cell
          int n_smaller_neighbors = 
            foreach_smaller_neighbor<ndim>( iCell_n, offset, U,
            [&]( const ForeachCell::CellIndex& iCell_sn )
          {
            auto pos = cells.getCellCenter(iCell_sn);
            auto size = cells.getCellSize(iCell_sn);
            res += pos + offset_real*size*0.5;
          });

          res /= n_smaller_neighbors;
        }
        else if( iCell_n.level_diff() == 1 )
        {
          auto pos_c = cells.getCellCenter(iCell);
          auto pos_n = cells.getCellCenter(iCell_n);
          auto size_n = cells.getCellSize(iCell_n);
          // Select the same-size virtual neighbor among biger cell's sectors
          StateNd<3> offset_quadrant;
          offset_quadrant[IX] = pos_n[IX] > pos_c[IX] ? -1. : +1.;
          offset_quadrant[IY] = pos_n[IY] > pos_c[IY] ? -1. : +1.;
          if( ndim == 3 )
            offset_quadrant[IZ] = pos_n[IZ] > pos_c[IZ] ? -1. : +1.;

          res = pos_n + offset_quadrant * size_n / 4;            
        }
      }        

      int nneighbors = (ndim==2)?4:6;
      U2.at( iCell, IU ) += res[IX]/nneighbors;
      U2.at( iCell, IV ) += res[IY]/nneighbors;
      U2.at( iCell, IW ) += res[IZ]/nneighbors;
    };

    append_offset({-1,+0,+0});
    append_offset({+1,+0,+0});
    append_offset({+0,-1,+0});
    append_offset({+0,+1,+0});
    if( ndim == 3 )
    {
      append_offset({+0,+0,-1});
      append_offset({+0,+0,+1});
    }
  });

  std::cout << "Check U2..." << std::endl;
  auto Uhost = Kokkos::create_mirror_view(U.U);
  Kokkos::deep_copy( Uhost, U.U );
  auto U2host = Kokkos::create_mirror_view(U2.U);
  Kokkos::deep_copy( Uhost, U.U );
  for( uint32_t iOct=0; iOct < amr_mesh->getNumOctants(); iOct++ )
    for( uint32_t i=0; i<bx*by*bz; i++ )
    {
      EXPECT_DOUBLE_EQ( U2host(i, iOct, 0 ), U2host(i, iOct, 0 ) );
      EXPECT_DOUBLE_EQ( U2host(i, iOct, 1 ), U2host(i, iOct, 1 ) );
      EXPECT_DOUBLE_EQ( U2host(i, iOct, 2 ), U2host(i, iOct, 2 ) );
    }

} // run_test



} // namespace dyablo

TEST(Test_ForeachCell, foreach_cell_mean_position_matches_center)
{
  dyablo::run_test();
}