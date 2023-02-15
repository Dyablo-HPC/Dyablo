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
    UserData::FieldAccessor U (U_, {{"px",Ipx},{"py",Ipy},{"pz",Ipz}});

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();
    foreach_cell.foreach_cell("Fill U", U.getShape(), 
        KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
        auto pos = cells.getCellCenter(iCell);
        U.at(iCell, Ipx) = pos[IX];
        U.at(iCell, Ipy) = pos[IY];
        U.at(iCell, Ipz) = pos[IZ];
    });

    int mpi_size = GlobalMpiSession::get_comm_world().MPI_Comm_size();
    Timers timers;
    configMap.getValue<std::string>("output", "outputDir", "./"),
    configMap.getValue<std::string>("output", "outputPrefix", std::string("test_")+std::to_string(mpi_size));
    std::unique_ptr<IOManager> io_checkpoint = IOManagerFactory::make_instance( "IOManager_checkpoint", configMap, foreach_cell, timers );
    
    std::cout << "Checkpoint...";
    io_checkpoint->save_snapshot(U_, 0, 0.0);
    std::cout << "Done" << std::endl;
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
    UserData::FieldAccessor U (U_, {{"px",Ipx},{"py",Ipy},{"pz",Ipz}});

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
}

} // namespace dyablo

TEST( test_restart, values_are_conserved )
{
    dyablo::test_checkpoint();
    dyablo::test_restart();
}

