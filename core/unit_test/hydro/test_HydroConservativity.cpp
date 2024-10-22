/**
 * Test all HydroUpdate implementations for conservativity
 * 
 * For each HydroUpdate registered in the factory int 2d/3d :
 * - Generate a low resolution blast with a few AMR levels in 2d / 3d,
 * - Execute HydroUpdate for a few iterations
 * - Check that the total mass and energy are conserved
 * 
 * Target conservativity for each HydroUpdate kernel can be 
 * personalized in expected_conservativity_percent()
 * Generate vizualization output files with the same name for 
 * each HydroUpdate/dim pair : use --gtest_filter to run a 
 * specific Kernel/dim and view the result
 **/

#include "utils/mpi/GlobalMpiSession.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "compute_dt/Compute_dt.h"
#include "init/InitialConditions.h"
#include "hydro/HydroUpdate.h"
#include "utils_hydro.h"
#include "io/IOManager.h"
#include "states/State_hydro.h"
#include "mpi/GhostCommunicator.h"
using blockSize_t    = Kokkos::Array<uint32_t, 3>;

using Device = Kokkos::DefaultExecutionSpace;

#include "gtest/gtest.h"

namespace dyablo {

real_t expected_conservativity_percent( const std::string& HydroUpdate_id )
{
  static std::map<std::string, real_t> expected_map =
  {
    {"HydroUpdate_hancock", 1},
    {"MHDUpdate_hancock", 1},
    {"HydroUpdate_legacy", 1},
    // Add custom conservativity target for non-conservative Solvers
  };
  real_t expected_default = 1e-10;

  auto it = expected_map.find( HydroUpdate_id );
  if(it == expected_map.end() )
    return expected_default;
  else
    return it->second;
}

using DiagArray = std::array<real_t, 2>;

struct DiagosticsFunctor {
  DiagosticsFunctor(ForeachCell& foreach_cell)
    : foreach_cell(foreach_cell) 
    {}

  DiagArray compute(const UserData &U_) {
    int ndim = foreach_cell.getDim();

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    enum VarIndex_diag{ ID, IE, IU, IV, IW };

    UserData::FieldAccessor U = U_.getAccessor( {
      {"rho", ID},
      {"e_tot", IE},
      {"rho_vx", IU},
      {"rho_vy", IV},
      {"rho_vz", IW}
    } );

    real_t mass = 0.0, energy = 0.0;
    foreach_cell.reduce_cell( "compute_diagnostics", U.getShape(),
    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& mass, real_t &energy)
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];
      real_t dV = dx*dy*(ndim==3 ? dz : 1.0);
      
      ConsHydroState uLoc{};
      uLoc.rho = U.at(iCell,ID);
      uLoc.e_tot = U.at(iCell,IE);
      uLoc.rho_u = U.at(iCell,IU);
      uLoc.rho_v = U.at(iCell,IV);
      uLoc.rho_w = (ndim==2)? 0 : U.at(iCell, IW);

      mass   += dV * uLoc.rho;
      energy += dV * uLoc.e_tot;

    }, Kokkos::Sum<real_t>(mass), Kokkos::Sum<real_t>(energy) );

    DiagArray res_local {mass, energy};
    DiagArray res;
    auto m_communicator = GlobalMpiSession::get_comm_world();
    m_communicator.MPI_Allreduce(res_local.data(), res.data(), 2, MpiComm::MPI_Op_t::SUM);
    return res;    
  }

  ForeachCell &foreach_cell;
};


std::shared_ptr<AMRmesh> init_amr_mesh( ConfigMap& configMap )
{
  int ndim = configMap.getValue<int>("mesh", "ndim", -1); // ndim was set prior to calling init_amr_mesh()
  int codim = ndim;
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
  int amr_level_min = configMap.getValue<int>("amr","level_min", 2);
  int amr_level_max = configMap.getValue<int>("amr","level_max", 4);
  return std::make_shared<AMRmesh>( ndim, codim, periodic, amr_level_min, amr_level_max );
}

void run_test(int ndim, std::string HydroUpdate_id ) {
  std::cout << "// =========================================\n";
  std::cout << "// Testing " << HydroUpdate_id << "\n";
  std::cout << "// =========================================\n";

  bool has_mhd = HydroUpdate_id.find("MHD") != std::string::npos;
  bool is_glm  = HydroUpdate_id.find("GLM") != std::string::npos;

  // Content of .ini file used to configure configmap and HydroParams
  std::string configmap_str;
  if (has_mhd) {
    configmap_str = 
        "[output]\n"
        "outputPrefix=test_Conservativity\n"
        "write_variables=rho,e_tot,Bx,By,Bz,level,rank\n"
        "enable_hdf5=on\n"
        "[amr]\n"
        "use_block_data=yes\n"
        "\n";
  }
  else {
    configmap_str = 
        "[output]\n"
        "outputPrefix=test_Conservativity\n"
        "write_variables=rho,e_tot,level,rank\n"
        "enable_hdf5=on\n"
        "[amr]\n"
        "use_block_data=yes\n"
        "\n";
  }

  ConfigMap configMap(configmap_str);

  // Set ndim in configmap
  configMap.getValue<int>("mesh", "ndim", ndim);
  // block sizes with ghosts
  configMap.getValue<uint32_t>("amr", "bx", 4);
  configMap.getValue<uint32_t>("amr", "by", 4);
  configMap.getValue<uint32_t>("amr", "bz", (ndim == 2 ? 1 : 4));

  auto amr_mesh = init_amr_mesh(configMap);
  ForeachCell foreach_cell(*amr_mesh, configMap);
  Timers timers;  
  
  // Initializing kernels
  std::cout << " . Initializing kernels" << std::endl;
  std::unique_ptr<Compute_dt> compute_dt;
  std::unique_ptr<HydroUpdate> updater;
  std::unique_ptr<IOManager> iomanager;
  {
    std::string compute_dt_id = configMap.getValue<std::string>("dt", "dt_kernel", "Compute_dt_hydro");
    compute_dt = Compute_dtFactory::make_instance("Compute_dt_hydro",
                                                  configMap,
                                                  foreach_cell, 
                                                  timers);

    updater = HydroUpdateFactory::make_instance(HydroUpdate_id, 
                                                configMap, 
                                                foreach_cell,
                                                timers);

    iomanager = IOManagerFactory::make_instance("IOManager_hdf5", 
                                                configMap, 
                                                foreach_cell,
                                                timers);
  }

  // Initializing data (U + amr_mesh)
  std::cout << " . Initializing data" << std::endl;
  UserData U( configMap, foreach_cell );
  {
    std::string init_name = configMap.getValue<std::string>("hydro", "problem", "blast");
    if (has_mhd)
      init_name = "MHD_" + init_name;

    auto initial_conditions = InitialConditionsFactory::make_instance(
                                init_name, 
                                configMap,
                                foreach_cell,
                                timers);

    initial_conditions->init(U);
  }

  int hydro_ghost_count;
  {          
    if( HydroUpdate_id.find("oneneighbor") != std::string::npos )
      hydro_ghost_count = 2; // Could be 1 but other kernels may need 2
    else if( HydroUpdate_id.find("hancock") != std::string::npos )
      hydro_ghost_count = 4;
    else
      hydro_ghost_count = 2;

    hydro_ghost_count = std::min( {U.getShape().bx, U.getShape().by, (uint32_t)hydro_ghost_count} );
  }

  GhostCommunicator ghost_comm(*amr_mesh, U.getShape(), hydro_ghost_count);

  auto exchange_ghosts = [&](const UserData& U)
  {  
    std::vector< UserData::FieldAccessor::FieldInfo > fields_to_exchange;
    int i=0;
    for( const std::string& field : U.getEnabledFields() )
    {
      fields_to_exchange.push_back( {field, i} );
      i++;
    }
    auto Uin = U.getAccessor( fields_to_exchange );
    ghost_comm.exchange_ghosts(Uin);
  };
  exchange_ghosts( U );


  DiagosticsFunctor diags(foreach_cell);
  {
    // Computing initial mass and energy
    std::cout << " . Computing initial quantities" << std::endl;
    DiagArray diag0 = diags.compute(U);
    
    ScalarSimulationData scalar_data;

    // Advancing a few steps
    std::cout << " . Running hydro solver" << std::endl;
    constexpr int nsteps = 10;
    real_t time = 0;
    scalar_data.set<int>("iter", 0);
    scalar_data.set<real_t>("time", time);
    iomanager->save_snapshot(U, scalar_data);
    for (int i=0; i < nsteps; ++i) {
      compute_dt->compute_dt(U, scalar_data);
      exchange_ghosts( U );

      // TODO automatic new fields according to kernel
      U.new_fields({"rho_next", "e_tot_next", "rho_vx_next", "rho_vy_next", "rho_vz_next"});
      if( has_mhd ) {
        U.new_fields({"Bx_next", "By_next", "Bz_next"});
        if ( is_glm ) {
          if (!U.has_field("psi"))
            U.new_fields({"psi"});
          U.new_fields({"psi_next"});
        }
      }

      updater->update(U, scalar_data); 

      U.move_field( "rho", "rho_next" ); 
      U.move_field( "e_tot", "e_tot_next" ); 
      U.move_field( "rho_vx", "rho_vx_next" ); 
      U.move_field( "rho_vy", "rho_vy_next" ); 
      U.move_field( "rho_vz", "rho_vz_next" );
      if( has_mhd )
      {
        U.move_field( "Bx", "Bx_next" ); 
        U.move_field( "By", "By_next" ); 
        U.move_field( "Bz", "Bz_next" );
        if ( is_glm )
          U.move_field( "psi", "psi_next" );
      }         
      
      time += scalar_data.get<real_t>("dt");
      scalar_data.set<int>("iter", i+1);
      scalar_data.set<real_t>("time",time);
      iomanager->save_snapshot(U, scalar_data);
      
      std::cout << "   Iteration #" << i << std::endl;
    }

    // 4- Computing final mass and energy
    std::cout << " . Computing final quantities" << std::endl;
    DiagArray diag1 = diags.compute(U);

    // 5- Checking values
    std::cout << " . Final conservation :" << std::endl;
    std::cout << "Quantity\tInitial\tFinal\tVariation\tPercent" << std::endl;
    real_t dM = diag1[0] - diag0[0];
    real_t dE = diag1[1] - diag0[1];
    real_t pct_M = 100.0*dM/diag0[0];
    real_t pct_E = 100.0*dE/diag0[1];
    std::cout << "Mass\t" << diag0[0] << "\t" << diag1[0] << "\t" << dM << "\t" << pct_M << std::endl;
    std::cout << "Energy\t" << diag0[1] << "\t" << diag1[1] << "\t" << dE << "\t" << pct_E << std::endl;

    real_t percent_threshold = expected_conservativity_percent(HydroUpdate_id);
    EXPECT_LT(fabs(pct_M), percent_threshold) << "Mass is not conserved at less than " << percent_threshold << "% !";
    EXPECT_LT(fabs(pct_E), percent_threshold) << "Energy is not conserved at less than " << percent_threshold << "% !";
  }

}

}

class Test_Conservativity 
  : public testing::TestWithParam<std::tuple<int, std::string>> 
{};

TEST_P(Test_Conservativity, mass_and_energy_conserved)
{
  int ndim = std::get<0>(GetParam());
  std::string id = std::get<1>(GetParam());
  dyablo::run_test(ndim, id );
}

INSTANTIATE_TEST_SUITE_P(
    Test_Conservativity, Test_Conservativity,
    testing::Combine(
        testing::Values(2,3),
        testing::ValuesIn( dyablo::HydroUpdateFactory::get_available_ids() )
    ),
    [](const testing::TestParamInfo<Test_Conservativity::ParamType>& info) {
      std::string name = 
          (std::get<0>(info.param) == 2 ? std::string("2D") : std::string("3D"))
          + "_" + std::get<1>(info.param);
      return name;
    }
);
