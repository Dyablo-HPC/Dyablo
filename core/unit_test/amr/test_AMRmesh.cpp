#include "gtest/gtest.h"

#include "amr/AMRmesh.h"
#include "utils/mpi/GlobalMpiSession.h"
#include "utils/monitoring/Timers.h"
#include "io/IOManager.h"

#include "foreach_cell/ForeachCell.h"

namespace dyablo {

struct Spot{
  real_t x, y, z; // At position {x,y,z}
  real_t r0;      // Refine at distance ( r0*cell_size ) for each level
  int level; // until this level
};

struct Test_data{
  int level_min;
  int level_max;
  real_t width;
};

template< typename AMRmesh_t >
void run_test(const Test_data& test_data)
{
  int dim = 3;
  bool perodic_x = true;
  bool perodic_y = true;
  bool perodic_z = true;
  int level_min = test_data.level_min;
  int level_max = test_data.level_max;
  real_t width = test_data.width;

  Timers timers;

  std::vector<Spot> spots = {
    {0.5,0.5,0.5,2,10},
    {0.25,0.25,0.25,width,level_max}
  };

  AMRmesh_t amr_mesh(dim, dim, {perodic_x,perodic_y,perodic_z}, level_min, level_max);
  if( test_data.level_max > amr_mesh.get_max_supported_level() )
  {
    std::cerr << "Test skipped : h=" << test_data.level_max << " unsupported by this AMRmesh implementation" << std::endl;
    GTEST_SKIP();
  }

  for( int i=0; i<level_min; i++ )
  {
    std::cout << "Global refine level " << i << std::endl;
    timers.get("adaptGlobalRefine").start();
    amr_mesh.adaptGlobalRefine();
    timers.get("adaptGlobalRefine").stop();
    // AMRmesh_hashmap cannot call adaptGlobalRefine after first loadBalance
    // timers.get("loadBalance").start();
    // amr_mesh.loadBalance();
    // timers.get("loadBalance").stop();
  }

  timers.get("loadBalance").start();
  amr_mesh.loadBalance();
  timers.get("loadBalance").stop();
  

  for( int level=level_min; level<level_max; level++ )
  {
    timers.get("MarkCells").start();
    std::cout << "Refine level " << level << std::endl;
    uint32_t nbOcts = amr_mesh.getNumOctants();
    int refine_count = 0;
    for(uint32_t iOct=0; iOct<nbOcts; iOct++)
    {
      auto c = amr_mesh.getCenter( iOct );
      real_t s = amr_mesh.getSize( iOct );
      bool refine = false;
      for( size_t i=0; i<spots.size(); i++ )
      {
        if( spots[i].level > level )
        {
          real_t dist_x = spots[i].x - c[0]; 
          real_t dist_y = spots[i].y - c[1]; 
          real_t dist_z = spots[i].z - c[2]; 

          real_t dist2 = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;
          real_t r2 = spots[i].r0*spots[i].r0*s*s;

          if( dist2 < r2  )
            refine = true;
        }
      }
      if(refine)
        refine_count++;
      amr_mesh.setMarker(iOct, refine?1:0);
    }
    std::cout << "Refine count " << refine_count << std::endl;
    timers.get("MarkCells").stop();
    timers.get("Adapt").start();
    amr_mesh.adapt();
    timers.get("Adapt").stop();
    timers.get("loadBalance").start();
    amr_mesh.loadBalance();
    timers.get("loadBalance").stop();    
  }

  uint32_t nbOcts = amr_mesh.getNumOctants();

  // Check total volume is 1
  {
    real_t V=0;
    for(int i=level_max; i>=level_min; i--)
    { // Iterate over levels to avoid rounding errors
      uint64_t count_level = 0;
      for(uint32_t iOct=0; iOct<nbOcts; iOct++)
      {
        real_t s = amr_mesh.getSize( iOct );
        int level = amr_mesh.getLevel( iOct );
        if( level==i )
        {
          V += s*s*s;
          count_level ++;
        }
      }
      std::cout << "level " << i << " : " << count_level << " octants" << std::endl;
    }
    real_t Vtot;
    GlobalMpiSession::get_comm_world().MPI_Allreduce(&V, &Vtot, 1, MpiComm::MPI_Op_t::SUM );
    
    EXPECT_DOUBLE_EQ( 1.0, Vtot );
  }

  if constexpr( std::is_same<AMRmesh_t, AMRmesh>::value )
  {
    // Output generated mesh
    char config_str[] = 
      "[output]\n"
      "hdf5_enabled=true\n"
      "write_mesh_info=true\n"
      "write_variables=\n"
      "write_iOct=false\n"
      "outputPrefix=output\n"
      "outputDir=./\n"
      "[amr]\n"
      "use_block_data=true\n"
      "bx=1\n"
      "by=1\n";
    char* config_str_ptr = config_str;
    ConfigMap configMap(config_str_ptr, strlen(config_str)); //Use default values
    ForeachCell foreach_cell( amr_mesh, configMap );
    ForeachCell::CellArray_global_ghosted U;
    IOManagerFactory::make_instance("IOManager_hdf5",configMap,foreach_cell,timers)->save_snapshot( U, 0, 0 );
  }

  // Verify 2:1 balance
  {
    dyablo::LightOctree_hashmap lmesh( &amr_mesh, amr_mesh.get_level_min(), amr_mesh.get_level_max() );

    int error_count = 0;
    Kokkos::parallel_reduce( "Check 2:1", nbOcts,
      KOKKOS_LAMBDA( uint32_t iOct, int& error_count_local )
      {
        for( int8_t nz=-1; nz<=1; nz++ )
          for( int8_t ny=-1; ny<=1; ny++ )
              for( int8_t nx=-1; nx<=1; nx++ )
                if( nx!=0 || ny!=0 || nz!=0 )
                {
                  int current_level = lmesh.getLevel({iOct, false});

                  bool boundary = lmesh.isBoundary( {iOct, false}, {nx,ny,nz} );
                  dyablo::LightOctree::NeighborList ns = lmesh.findNeighbors( {iOct, false}, {nx,ny,nz} );
                  //EXPECT_TRUE( ns.size()>0 || boundary );

                  for( int i=0; i<ns.size(); i++ )
                  {
                    int neighbor_level = lmesh.getLevel(ns[i]);
                    //EXPECT_TRUE( std::abs(neighbor_level - current_level ) <= 1 );
                    if( abs(neighbor_level - current_level ) > 1 )
                      error_count_local++;
                  }
                }
      }, error_count);

    EXPECT_EQ( 0, error_count );

  }


  timers.print();
}

} // namespace dyablo

using namespace dyablo;

template< typename AMRmesh_t_ >
class Test_AMRmesh
  : public testing::Test
{
public:
  using AMRmesh_t = AMRmesh_t_;
};

using AMRmesh_types = ::testing::Types<AMRmesh_pablo, AMRmesh_hashmap, AMRmesh_hashmap_new>;
TYPED_TEST_SUITE( Test_AMRmesh, AMRmesh_types );

TYPED_TEST(Test_AMRmesh, narrow_h6)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 6;
  td.width = 5;
  run_test<dyablo::AMRmesh_impl<typename TestFixture::AMRmesh_t>>(td);
}

TYPED_TEST(Test_AMRmesh, narrow_h18)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 18;
  td.width = 5;
  run_test<dyablo::AMRmesh_impl<typename TestFixture::AMRmesh_t>>(td);
}

TYPED_TEST(Test_AMRmesh, narrow_h20)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 20;
  td.width = 5;
  run_test<dyablo::AMRmesh_impl<typename TestFixture::AMRmesh_t>>(td);
}

TYPED_TEST(Test_AMRmesh, wide_h10)
{
  Test_data td{};
  td.level_min = 4;
  td.level_max = 10;
  td.width = 15;
  run_test<dyablo::AMRmesh_impl<typename TestFixture::AMRmesh_t>>(td);
}