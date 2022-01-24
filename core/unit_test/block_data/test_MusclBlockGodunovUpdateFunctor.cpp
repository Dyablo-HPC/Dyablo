/**
 * \file test_MusclBlockGodunovUpdateFunctor.cpp
 * \author Pierre Kestener
 * \date October, 1st 2019
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>


#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/FieldManager.h"

#include "muscl_block/init/InitialConditions.h"

#include "muscl_block/update/MusclBlockUpdate.h"
#include "muscl_block/legacy/ComputeDtHydroFunctor.h"

using Device = Kokkos::DefaultExecutionSpace;

#include <boost/test/unit_test.hpp>

namespace dyablo {

namespace muscl_block {

void run_test(std::string name, std::string filename) {

  std::cout << "// =========================================\n";
  std::cout << "// Testing " << name << " ...\n";
  std::cout << "// =========================================\n";

  /*
   * read parameter file and initialize a ConfigMap object
   */
  // only MPI rank 0 actually reads input file
  //std::string input_file = std::string(argv[1]);
  std::string input_file = filename;
  ConfigMap configMap = ConfigMap::broadcast_parameters(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  Timers timers;

  int ndim = params.dimType == TWO_D ? 2 : 3;
  // block sizes
  uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 0);
  uint32_t by = configMap.getValue<uint32_t>("amr", "by", 0);
  uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", 1);
  FieldManager field_manager({ID,IP,IU,IV,IW});
  AMRmesh amr_mesh( ndim, ndim, {false, false, false}, 2, 4 );  

  std::string init_name = configMap.getValue<std::string>("hydro", "problem", "unknown");

  ForeachCell foreach_cell(amr_mesh, configMap);

  ForeachCell::CellArray_global_ghosted U;
  // Initialize cells
  {    
    std::unique_ptr<InitialConditions> initial_conditions =
      InitialConditionsFactory::make_instance(init_name, 
        configMap,
        foreach_cell,
        timers);

    initial_conditions->init(U, field_manager);
  }
  ForeachCell::CellArray_global_ghosted U2 = foreach_cell.allocate_ghosted_array("U2", field_manager); 
  
  auto Uhost = Kokkos::create_mirror_view(U.U);

  // by now, init condition must have been called

  auto fm = field_manager.get_id2index();

  /*
   * "geometry" setup
   */

  // block ghost width
  uint32_t ghostWidth = configMap.getValue<uint32_t>("amr", "ghostwidth", 2);

  // block sizes with ghosts
  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = by + 2 * ghostWidth;
  uint32_t bz_g = bz + 2 * ghostWidth;

  blockSize_t blockSizes, blockSizes_g;
  blockSizes[IX] = bx;
  blockSizes[IY] = by;
  blockSizes[IZ] = bz;

  blockSizes_g[IX] = bx_g;
  blockSizes_g[IY] = by_g;
  blockSizes_g[IZ] = bz_g;

  std::cout << "Using " 
            << "bx=" << bx << " "
            << "by=" << by << " "
            << "bz=" << bz << " "
            << "bx_g=" << bx_g << " "
            << "by_g=" << by_g << " "
            << "bz_g=" << bz_g << " "
            << "ghostwidth=" << ghostWidth << "\n";

  uint32_t nbCellsPerOct_g =
      ndim == 2 ? bx_g * by_g : bx_g * by_g * bz_g;

  uint32_t nbOctsPerGroup = configMap.getValue<uint32_t>("amr", "nbOctsPerGroup", 32);

  /*
   * allocate/initialize Ugroup / Qgroup
   */

  uint32_t nbOcts = amr_mesh.getNumOctants();
  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";
  
  uint32_t iGroup = 1;

 // // chose an octant which should have a "same size" neighbor in all direction
  // //uint32_t iOct_local = 2;
  
  // // chose an octant which should have at least
  // // a "larger size" neighbor in one direction
  // //uint32_t iOct_local = 30;

  // // chose an octant which should have at least
  // // an interface with "smaller size" neighbor in one direction
  uint32_t iOct_local = 26;

  uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;

  // std::cout << "Looking at octant id = " << iOct_global << "\n";

  // // save solution, just for cross-checking
  // save_solution();

  std::cout << "Printing U data from iOct = " << iOct_global << "\n";
  for (uint32_t iz=0; iz<bz; ++iz) {
    for (uint32_t iy=0; iy<by; ++iy) {
      for (uint32_t ix=0; ix<bx; ++ix) {
        uint32_t index = ix + bx*(iy+by*iz);
        printf("%5f ", Uhost(index,fm[ID],iOct_global));
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  // first copy inner cells

  params.gravity_type = GRAVITY_NONE;

  // compute CFL constraint
  real_t invDt;
  ComputeDtHydroFunctor::apply(amr_mesh.getLightOctree(),
                               ComputeDtHydroFunctor::Params(configMap),
                               fm,
                               blockSizes,
                               U.U,
                               invDt);  
  real_t dt = params.settings.cfl / invDt;

  printf("CFL dt = %f\n",dt);

  // testing MusclBlockGodunovUpdateFunctor
  {
    std::unique_ptr<MusclBlockUpdate> godunov_updater = MusclBlockUpdateFactory::make_instance( "MusclBlockUpdate_generic",
      configMap,
      foreach_cell,
      timers
    );
    godunov_updater->update(U, U2, dt);

    Kokkos::deep_copy( Uhost, U.U);
    std::cout << "Printing U data (after update) from iOct = " << iOct_global << "\n";
    for (uint32_t iy = 0; iy < by; ++iy) {
      for (uint32_t ix = 0; ix < bx; ++ix) {
        uint32_t index = ix + bx * iy;
        printf("%5f ", Uhost(index, fm[ID], iOct_global));
      }
      std::cout << "\n";
    }
    std::cout << "\n";
    
    DataArrayBlockHost U2_host = Kokkos::create_mirror_view(U2.U);
    std::cout << "Printing U2 data (after update) from iOct = " << iOct_global << "\n";
    Kokkos::deep_copy( U2_host, U2.U);
    for (uint32_t iy = 0; iy < by; ++iy) {
      for (uint32_t ix = 0; ix < bx; ++ix) {
        uint32_t index = ix + bx * iy;
        printf("%5f ", U2_host(index, fm[ID], iOct_global));
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

} // run_test

} // namespace muscl_block

} // namespace dyablo


BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_SUITE(muscl_block)

BOOST_AUTO_TEST_CASE(test_MusclBlockUpdate_generic_blast_2D)
{
  run_test("MusclBlockUpdate_generic (2D)", "./block_data/test_blast_2D_block.ini");
}

BOOST_AUTO_TEST_CASE(test_MusclBlockUpdate_generic_blast_3D)
{
  run_test("MusclBlockUpdate_generic (3D)", "./block_data/test_blast_3D_block.ini");
}

BOOST_AUTO_TEST_SUITE_END() /* muscl_block */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */

