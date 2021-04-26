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
#include "shared/solver_utils.h" // print monitoring information
#include "shared/FieldManager.h"

#ifdef DYABLO_USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

#include "muscl_block/SolverHydroMusclBlock.h"

#include "muscl_block/update/MusclBlockUpdate.h"
#include "muscl_block/ComputeDtHydroFunctor.h"

using Device = Kokkos::DefaultExecutionSpace;

#include <boost/test/unit_test.hpp>

namespace dyablo {

namespace muscl_block {

void run_test(int argc, char *argv[]) {

  std::cout << "// =========================================\n";
  std::cout << "// Testing MusclBlockUpdate (2D) ...\n";
  std::cout << "// =========================================\n";

  /*
   * read parameter file and initialize a ConfigMap object
   */
  // only MPI rank 0 actually reads input file
  //std::string input_file = std::string(argv[1]);
  std::string input_file = argc>1 ? std::string(argv[1]) : "./block_data/test_implode_2D_block.ini";
  ConfigMap configMap = broadcast_parameters(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // actually initializing a solver
  // initialize workspace memory (U, U2, ...)
  if (solver_name.find("Muscl_Block") == std::string::npos) {

    std::cerr << "Please modify your input parameter file.\n";
    std::cerr << "  solver name must contain string \"Muscl_Block\"\n";

  }

  // anyway create the right solver
  std::unique_ptr<SolverHydroMusclBlock> solver = std::make_unique<SolverHydroMusclBlock>(params, configMap);

  // by now, init condition must have been called

  // just retrieve a field manager
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);

  auto fm = fieldMgr.get_id2index();

  /*
   * "geometry" setup
   */

  // block ghost width
  uint32_t ghostWidth = configMap.getInteger("amr", "ghostwidth", 2);

  // block sizes
  uint32_t bx = configMap.getInteger("amr", "bx", 0);
  uint32_t by = configMap.getInteger("amr", "by", 0);
  uint32_t bz = configMap.getInteger("amr", "bz", 1);

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

  //uint32_t nbCellsPerOct = params.dimType == TWO_D ? bx * by : bx * by * bz;
  uint32_t nbCellsPerOct_g =
      params.dimType == TWO_D ? bx_g * by_g : bx_g * by_g * bz_g;

  uint32_t nbOctsPerGroup = configMap.getInteger("amr", "nbOctsPerGroup", 32);

  /*
   * allocate/initialize Ugroup / Qgroup
   */

  uint32_t nbOcts = solver->amr_mesh->getNumOctants();
  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";
  
  uint32_t iGroup = 1;

  uint8_t nfaces = (params.dimType == TWO_D ? 4 : 6);
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
  // solver->save_solution();

  std::cout << "Printing U data from iOct = " << iOct_global << "\n";
  for (uint32_t iz=0; iz<bz; ++iz) {
    for (uint32_t iy=0; iy<by; ++iy) {
      for (uint32_t ix=0; ix<bx; ++ix) {
        uint32_t index = ix + bx*(iy+by*iz);
        printf("%5f ",solver->Uhost(index,fm[ID],iOct_global));
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  // first copy inner cells

  params.gravity_type = GRAVITY_NONE;

  //uint32_t nbOcts = solver->amr_mesh->getNumOctants();

  LightOctree lmesh(solver->amr_mesh,params.level_min,params.level_max);

  // compute CFL constraint
  real_t invDt;
  ComputeDtHydroFunctor::apply(lmesh,
                               configMap,
                               params,
                               fm,
                               blockSizes,
                               solver->U,
                               invDt);  
  real_t dt = params.settings.cfl / invDt;

  printf("CFL dt = %f\n",dt);

  // testing MusclBlockGodunovUpdateFunctor
  {
    std::unique_ptr<MusclBlockUpdate> godunov_updater = MusclBlockUpdateFactory::make_instance( "MusclBlockUpdate_legacy",
      configMap,
      params,
      lmesh, 
      fm,
      bx, by, bz,
      solver->timers
    );
    godunov_updater->update(solver->U, solver->Ughost, solver->U2, dt);

    Kokkos::deep_copy( solver->Uhost, solver->U);
    std::cout << "Printing U data (after update) from iOct = " << iOct_global << "\n";
    for (uint32_t iy = 0; iy < by; ++iy) {
      for (uint32_t ix = 0; ix < bx; ++ix) {
        uint32_t index = ix + bx * iy;
        printf("%5f ", solver->Uhost(index, fm[ID], iOct_global));
      }
      std::cout << "\n";
    }
    std::cout << "\n";
    
    DataArrayBlockHost U2_host = Kokkos::create_mirror_view(solver->U2);
    std::cout << "Printing U2 data (after update) from iOct = " << iOct_global << "\n";
    Kokkos::deep_copy( U2_host, solver->U2);
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

BOOST_AUTO_TEST_CASE(test_MusclBlockGodunovUpdateFunctor)
{

  run_test(boost::unit_test::framework::master_test_suite().argc,
           boost::unit_test::framework::master_test_suite().argv);

}

BOOST_AUTO_TEST_SUITE_END() /* muscl_block */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */

