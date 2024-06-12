/**
 * Test the checkpoint/restart mechanism
 * - Initialize mesh with positions in cells
 * - Perform checkpoint
 * - Load mesh from restart 
 * - Check loaded mesh is identical
 **/

#include "gtest/gtest.h"

#include "amr/AMRmesh.h"
#include "config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"
#include "io/IOManager.h"
#include "init/InitialConditions.h"

namespace dyablo {

uint32_t expected_oct_count = 701;

/// Save checkpoint file with positions in cells
void test_checkpoint()
{
    // Initialize AMRmesh
    std::cout << "Create mesh..." << std::endl;

    // Content of .ini file used to configure configmap 
    std::string configmap_str = R"ini(
[amr]
level_min=3
level_max=5
bx=4
by=4
)ini";
    ConfigMap configMap(configmap_str);

    int ndim = 3;
    configMap.getValue<int>("mesh", "ndim", ndim);
    configMap.getValue<uint32_t>("amr", "bz", (ndim == 2 ? 1 : 4));

    // Initialize AMRmesh
    std::shared_ptr<AMRmesh> amr_mesh;
    {
        // Get/initialize parameters in configmap
        BoundaryConditionType bxmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_PERIODIC);
        BoundaryConditionType bxmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_PERIODIC);
        BoundaryConditionType bymin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_PERIODIC);
        BoundaryConditionType bymax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_PERIODIC);
        BoundaryConditionType bzmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_PERIODIC);
        BoundaryConditionType bzmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_PERIODIC);
        std::array<bool,3> periodic = {
            bxmin == BC_PERIODIC || bxmax == BC_PERIODIC,
            bymin == BC_PERIODIC || bymax == BC_PERIODIC,
            bzmin == BC_PERIODIC || bzmax == BC_PERIODIC
        };
        int level_min = configMap.getValue<int>("amr","level_min", 3);
        int level_max = configMap.getValue<int>("amr","level_max", 5);

        // Create initial mesh
        amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, periodic, level_min, level_max);

        // refine one level
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
        amr_mesh->loadBalance(0);
    }

    EXPECT_EQ( expected_oct_count , amr_mesh->getGlobalNumOctants() );

    // Fill U with positions
    ForeachCell foreach_cell( *amr_mesh, configMap );
    UserData U_ ( configMap, foreach_cell );
    U_.new_fields({"px","py","pz"});
    enum VarIndex_checkpoint {Ipx, Ipy, Ipz};
    UserData::FieldAccessor U = U_.getAccessor( {{"px",Ipx},{"py",Ipy},{"pz",Ipz}});

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();
    foreach_cell.foreach_cell("Fill U", U.getShape(), 
        KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
        auto pos = cells.getCellCenter(iCell);
        U.at(iCell, Ipx) = pos[IX];
        U.at(iCell, Ipy) = pos[IY];
        U.at(iCell, Ipz) = pos[IZ];
    });

    ForeachParticle foreach_particle( *amr_mesh, configMap  );

    uint32_t Wp1 = 10;
    uint32_t Np1 = Wp1*Wp1*Wp1;
    U_.new_ParticleArray("p1", Np1);
    U_.new_ParticleAttribute("p1", "a1");
    U_.new_ParticleAttribute("p1", "a2");

    const auto& P1 = U_.getParticleArray("p1");
    enum VarIndex_chekpoint_p1{IA1, IA2};
    UserData::ParticleAccessor P1a = U_.getParticleAccessor( "p1", {{"a1",IA1}, {"a2",IA2}} );
    foreach_particle.foreach_particle( "Fill p1", U_.getParticleArray("p1"),
        KOKKOS_LAMBDA(const ForeachParticle::ParticleIndex& iPart)
    {
      uint32_t ix =  iPart%Wp1;
      uint32_t iy = (iPart/Wp1)%Wp1;
      uint32_t iz =  iPart/(Wp1*Wp1);

      P1.pos( iPart, IX ) = (ix+0.5)/Wp1;
      P1.pos( iPart, IY ) = (iy+0.5)/Wp1;
      P1.pos( iPart, IZ ) = (ndim-2)*((iz+0.5)/Wp1);
      P1a.at( iPart, IA1 ) = ix;
      P1a.at( iPart, IA2 ) = iy;
    });


    int mpi_size = GlobalMpiSession::get_comm_world().MPI_Comm_size();
    Timers timers;
    configMap.getValue<std::string>("output", "outputDir", "./"),
    configMap.getValue<std::string>("output", "outputPrefix", std::string("test_")+std::to_string(mpi_size));
    std::unique_ptr<IOManager> io_checkpoint = IOManagerFactory::make_instance( "IOManager_checkpoint", configMap, foreach_cell, timers );
    
    ScalarSimulationData scalar_data;
    scalar_data.set<int>("iter", 0);
    scalar_data.set<real_t>("time",1.0);

    std::cout << "Checkpoint...";
    io_checkpoint->save_snapshot(U_, scalar_data);
    std::cout << "Done" << std::endl;
    
    configMap.getValue<std::string>("output", "write_particle_variables", "p1/a1,p1/a2" );
    std::unique_ptr<IOManager> io_hdf5 = IOManagerFactory::make_instance( "IOManager_hdf5", configMap, foreach_cell, timers );
    std::cout << "Output...";
    io_hdf5->save_snapshot(U_, scalar_data);
    std::cout << "Done" << std::endl;
}

// Shamelessly stolen from google test and adapted for Kokkos device code
// https://github.com/google/googletest/blob/a7f443b80b105f940225332ed3c31f2790092f47/googletest/include/gtest/internal/gtest-internal.h#L336
// Returns true if and only if v1 is at most kMaxUlps=4 ULP's away
// from v2.  In particular, this function:
//
//   - returns false if either number is (or both are) NAN.
//   - treats really large numbers as almost equal to infinity.
//   - thinks +0.0 and -0.0 are 0 DLP's apart.
template<typename RawType>
KOKKOS_INLINE_FUNCTION
bool almost_equal(RawType v1, RawType v2)
{
  using Bits = typename testing::internal::TypeWithSize<sizeof(RawType)>::UInt;

  if (isnan(v1) || isnan(v2)) return false;

  auto SignAndMagnitudeToBiased = [](const Bits &sam) {
    Bits kSignBitMask = static_cast<Bits>(1) << (8*sizeof(RawType) - 1);
    if (kSignBitMask & sam) {
      // sam represents a negative number.
      return ~sam + 1;
    } else {
      // sam represents a positive number.
      return kSignBitMask | sam;
    }
  };
  
  Bits* sam1 = reinterpret_cast<Bits*>(&v1);
  Bits* sam2 = reinterpret_cast<Bits*>(&v2);
  const Bits biased1 = SignAndMagnitudeToBiased(*sam1);
  const Bits biased2 = SignAndMagnitudeToBiased(*sam2);
  uint32_t kMaxUlps = 4;
  return ( (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1) ) <= kMaxUlps;
}

/// Load previously saved checkpoint and check values
void test_restart()
{
    std::cout << "Restart..." << std::endl;

    int mpi_size = GlobalMpiSession::get_comm_world().MPI_Comm_size();
    std::string ini_filename = std::string("restart_test_")+std::to_string(mpi_size)+"_0.ini";
    std::cout << "Load " << ini_filename << std::endl;
    ConfigMap configMap = ConfigMap::broadcast_parameters( ini_filename );

    int ndim = configMap.getValue<int>("mesh", "ndim", -1);
    int codim = ndim;
    BoundaryConditionType bxmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_UNDEFINED);
    BoundaryConditionType bxmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_UNDEFINED);
    BoundaryConditionType bymin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_UNDEFINED);
    BoundaryConditionType bymax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_UNDEFINED);
    BoundaryConditionType bzmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_UNDEFINED);
    BoundaryConditionType bzmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_UNDEFINED);
    std::array<bool,3> periodic = {
      bxmin == BC_PERIODIC || bxmax == BC_PERIODIC,
      bymin == BC_PERIODIC || bymax == BC_PERIODIC,
      bzmin == BC_PERIODIC || bzmax == BC_PERIODIC
    };
    int amr_level_min = configMap.getValue<int>("amr","level_min", -1);
    int amr_level_max = configMap.getValue<int>("amr","level_max", -1);
    AMRmesh amr_mesh( ndim, codim, periodic, amr_level_min, amr_level_max );

    ForeachCell foreach_cell( amr_mesh, configMap );
    UserData U_ ( configMap, foreach_cell );

    Timers timers;

    // Initialize cells
    std::unique_ptr<InitialConditions> initial_conditions =
      InitialConditionsFactory::make_instance("restart", 
        configMap,
        foreach_cell,
        timers);
    initial_conditions->init( U_);

    EXPECT_EQ( expected_oct_count , amr_mesh.getGlobalNumOctants() );

    enum VarIndex_restart {Ipx, Ipy, Ipz};
    UserData::FieldAccessor U = U_.getAccessor( {{"px",Ipx},{"py",Ipy},{"pz",Ipz}});

    int error_count = 0;
    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();
    foreach_cell.reduce_cell("Test U", U.getShape(), 
        KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell, int& err)
    {
        auto pos = cells.getCellCenter(iCell);
        if( U.at(iCell, Ipx) != pos[IX] )
            err ++;
        if( U.at(iCell, Ipy) != pos[IY] )
            err ++;
        if( U.at(iCell, Ipz) != pos[IZ] )
            err ++;
    }, error_count);

    EXPECT_EQ( 0, error_count );
    error_count=0;

    uint32_t Wp1 = 10;
    uint32_t Np1 = Wp1*Wp1*Wp1;

    EXPECT_TRUE( U_.has_ParticleArray("p1") );
    EXPECT_TRUE( U_.has_ParticleAttribute("p1", "a1") );
    EXPECT_TRUE( U_.has_ParticleAttribute("p1", "a2") );

    const auto& P1 = U_.getParticleArray("p1");
    EXPECT_EQ( P1.getNumParticles() , Np1 );

    ForeachParticle foreach_particle( amr_mesh, configMap  );
    enum VarIndex_chekpoint_p1{IA1, IA2};
    UserData::ParticleAccessor P1a = U_.getParticleAccessor( "p1", {{"a1",IA1}, {"a2",IA2}} );
    foreach_particle.reduce_particle( "check p1", U_.getParticleArray("p1"),
        KOKKOS_LAMBDA(const ForeachParticle::ParticleIndex& iPart, int& error_count)
    {
      uint32_t ix =  iPart%Wp1;
      uint32_t iy = (iPart/Wp1)%Wp1;
      uint32_t iz =  iPart/(Wp1*Wp1);

      auto expect_equal = [&](real_t expected, real_t actual)
      {
        //EXPECT_DOUBLE_EQ(expected, actual);       
        if( !almost_equal( expected, actual ) )
          error_count++;
      };

      expect_equal( P1.pos( iPart, IX ), (ix+0.5)/Wp1);
      expect_equal( P1.pos( iPart, IY ), (iy+0.5)/Wp1);
      expect_equal( P1.pos( iPart, IZ ), (ndim-2)*((iz+0.5)/Wp1));
      expect_equal( P1a.at( iPart, IA1 ), ix);
      expect_equal( P1a.at( iPart, IA2 ), iy);
    }, error_count);

    EXPECT_EQ( 0, error_count );


}

} // namespace dyablo

TEST( test_restart, values_are_conserved )
{
    dyablo::test_checkpoint();
    dyablo::test_restart();
}

