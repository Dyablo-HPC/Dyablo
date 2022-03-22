
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>


#include "real_type.h"    // choose between single and double precision
#include "FieldManager.h"

#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "compute_dt/Compute_dt.h"
#include "init/InitialConditions.h"
#include "hydro/HydroUpdate.h"
#include "utils_hydro.h"
using blockSize_t    = Kokkos::Array<uint32_t, 3>;

using Device = Kokkos::DefaultExecutionSpace;

#include "gtest/gtest.h"

namespace dyablo {

using DiagArray = std::array<real_t, 2>;

struct DiagosticsFunctor {
  DiagosticsFunctor(ForeachCell& foreach_cell)
    : foreach_cell(foreach_cell) 
    {}

  DiagArray compute(const ForeachCell::CellArray_global_ghosted &U) {
    int ndim = foreach_cell.getDim();

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t mass = 0.0, energy = 0.0;
    foreach_cell.reduce_cell( "compute_diagnostics", U,
    CELL_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& mass, real_t &energy)
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];
      real_t dV = dx*dy*(ndim==3 ? dz : 1.0);
      
      HydroState3d uLoc;
      uLoc[ID] = U.at(iCell,ID);
      uLoc[IP] = U.at(iCell,IP);
      uLoc[IU] = U.at(iCell,IU);
      uLoc[IV] = U.at(iCell,IV);
      uLoc[IW] = (ndim==2)? 0 : U.at(iCell, IW);

      mass   += dV * uLoc[ID];
      energy += dV * uLoc[IE];

    }, Kokkos::Sum<real_t>(mass), Kokkos::Sum<real_t>(energy) );

    DiagArray res {mass, energy};
    return res;    
  }

  ForeachCell &foreach_cell;
};


std::shared_ptr<AMRmesh> init_amr_mesh( ConfigMap& configMap )
{
  int ndim = configMap.getValue<int>("mesh", "ndim", 3);
  int codim = ndim;
  BoundaryConditionType bxmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING);
  BoundaryConditionType bxmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING);
  BoundaryConditionType bymin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING);
  BoundaryConditionType bymax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING);
  BoundaryConditionType bzmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING);
  BoundaryConditionType bzmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING);
  std::array<bool,3> periodic = {
    bxmin == BC_PERIODIC || bxmax == BC_PERIODIC,
    bymin == BC_PERIODIC || bymax == BC_PERIODIC,
    bzmin == BC_PERIODIC || bzmax == BC_PERIODIC
  };
  int amr_level_min = configMap.getValue<int>("amr","level_min", 5);
  int amr_level_max = configMap.getValue<int>("amr","level_max", 10);
  return std::make_shared<AMRmesh>( ndim, codim, periodic, amr_level_min, amr_level_max );
}

template<int ndim>
void run_test() {

  std::cout << "// =========================================\n";
  std::cout << "// Testing conservativity ...\n";
  std::cout << "// =========================================\n";
  
  std::string input_file = (ndim == 2 ? "./hydro/test_blast_2D_block.ini" : "./hydro/test_blast_3D_block.ini");
  ConfigMap configMap = ConfigMap::broadcast_parameters(input_file);
  std::string init_name = configMap.getValue<std::string>("hydro", "problem", "unknown");

  FieldManager field_manager = (ndim == 2 
                              ? FieldManager({ID, IP, IU, IV}) 
                              : FieldManager({ID, IP, IU, IV, IW}));

  // block ghost width
  uint32_t ghostWidth = configMap.getValue<uint32_t>("amr", "ghostwidth", 2);

  // block sizes with ghosts
  uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 4);
  uint32_t by = configMap.getValue<uint32_t>("amr", "by", 4);
  uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", 4);
  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = by + 2 * ghostWidth;
  uint32_t bz_g = bz + 2 * ghostWidth;

  blockSize_t blockSizes, blockSizes_g;
  blockSizes[IX] = bx;
  blockSizes[IY] = by;
  blockSizes[IZ] = (ndim == 2 ? 1 : bz);

  blockSizes_g[IX] = bx_g;
  blockSizes_g[IY] = by_g;
  blockSizes_g[IZ] = (ndim == 2 ? 1 : bz_g);


  auto amr_mesh = init_amr_mesh(configMap);
  ForeachCell foreach_cell(*amr_mesh, configMap);
  Timers timers;    
  
  std::string compute_dt_id = configMap.getValue<std::string>("dt", "dt_kernel", "Compute_dt_generic");
  auto compute_dt = Compute_dtFactory::make_instance( compute_dt_id,
      configMap,
      foreach_cell, 
      timers
  );
  
  ForeachCell::CellArray_global_ghosted U;
  auto Uhost = Kokkos::create_mirror_view(U.U);

  auto ids = HydroUpdateFactory::get_available_ids();
  DiagosticsFunctor diags(foreach_cell);
  for (auto &k : ids) {
    if (k == "HydroUpdate_muscl_oneneighbor") {
      std::cout << "Skipping " << k << std::endl;
      continue;
    }

    DiagArray diag0;
    DiagArray diag1;

    std::cout << std::endl << std::endl;
    std::cout << "==== Testing construct name : " << k << std::endl;
    auto updater = HydroUpdateFactory::make_instance(k, configMap, 
                                                  foreach_cell,
                                                  timers);

    auto initial_conditions =
      InitialConditionsFactory::make_instance(init_name, 
        configMap,
        foreach_cell,
        timers);

    // 1- Initializing data
    std::cout << " . Initializing" << std::endl;
    initial_conditions->init(U, field_manager);
    ForeachCell::CellArray_global_ghosted U2 = foreach_cell.allocate_ghosted_array("U2", field_manager); 

    // 2- Computing initial mass and energy
    std::cout << " . Computing initial quantities" << std::endl;
    diag0 = diags.compute(U);

    // 3- Advancing a few steps
    std::cout << " . Running hydro solver" << std::endl;
    constexpr int nsteps = 10;
    for (int i=0; i < nsteps; ++i) {
      real_t dt = compute_dt->compute_dt(U);
      std::cout << "   Iteration #" << i << std::endl;
      updater->update(U, U2, dt);
      Kokkos::deep_copy(U.U, U2.U);
    }

    // 4- Computing final mass and energy
    std::cout << " . Computing final quantities" << std::endl;
    diag1 = diags.compute(U);

    // 5- Checking values
    std::cout << " . Final conservation :" << std::endl;
    std::cout << "Quantity\tInitial\tFinal\tVariation\tPercent" << std::endl;
    real_t dM = diag1[0] - diag0[0];
    real_t dE = diag1[1] - diag1[1];
    real_t pct_M = 100.0*dM/diag0[0];
    real_t pct_E = 100.0*dE/diag0[1];
    std::cout << "Mass\t" << diag0[0] << "\t" << diag1[0] << "\t" << dM << "\t" << pct_M << std::endl;
    std::cout << "Energy\t" << diag0[1] << "\t" << diag1[1] << "\t" << dE << "\t" << pct_E << std::endl;

    EXPECT_LT(fabs(pct_M), 0.1) << "Mass is not conserved at less than 0.1% !";
    EXPECT_LT(fabs(pct_E), 0.1) << "Energy is not conserved at less than 0.1% !";
  }

}

}

TEST(dyablo, test_Conservativity_2D)
{
  dyablo::run_test<2>();
}

TEST(dyablo, test_Conservativity_3D)
{
  dyablo::run_test<3>();
}
