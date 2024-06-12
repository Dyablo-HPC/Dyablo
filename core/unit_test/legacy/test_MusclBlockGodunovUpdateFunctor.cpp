/**
 * \file test_MusclBlockGodunovUpdateFunctor.cpp
 * \author Pierre Kestener
 * \date October, 1st 2019
 */

#include <Kokkos_Core.hpp>

#include "real_type.h"    // choose between single and double precision
#include "FieldManager.h"

#include "init/InitialConditions.h"

#include "hydro/HydroUpdate.h"
#include "legacy/ComputeDtHydroFunctor.h"

using Device = Kokkos::DefaultExecutionSpace;

#include "gtest/gtest.h"

namespace dyablo {


void run_test(std::string name, int ndim) {

  std::cout << "// =========================================\n";
  std::cout << "// Testing " << name << " ...\n";
  std::cout << "// =========================================\n";

  // Content of .ini file used ton configure configmap and HydroParams
  std::string configmap_str = 
    "[mesh]\n"
    "boundary_type_xmin=absorbing\n"
    "boundary_type_xmax=absorbing\n"
    "boundary_type_ymin=absorbing\n"
    "boundary_type_ymax=absorbing\n"
    "boundary_type_zmin=absorbing\n"
    "boundary_type_zmax=absorbing\n"
    "[gravity]\n"
    "gravity_type=none\n"
    "\n";
  ConfigMap configMap(configmap_str);

  Timers timers;

  // block sizes
  configMap.getValue<int>("mesh", "ndim", ndim);
  uint32_t bx = configMap.getValue<uint32_t>("amr", "bx",4);
  uint32_t by = configMap.getValue<uint32_t>("amr", "by",4);
  uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", (ndim==3)?4:1);
  FieldManager field_manager({ID,IP,IU,IV,IW});
  AMRmesh amr_mesh( ndim, ndim, {false, false, false}, 2, 4 );  

  std::string init_name = "blast";

  ForeachCell foreach_cell(amr_mesh, configMap);

  UserData U_(configMap, foreach_cell);
  // Initialize cells
  {    
    std::unique_ptr<InitialConditions> initial_conditions =
      InitialConditionsFactory::make_instance(init_name, 
        configMap,
        foreach_cell,
        timers);

    initial_conditions->init(U_);
  }

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

  uint32_t nbOctsPerGroup = configMap.getValue<uint32_t>("amr", "nbOctsPerGroup", 1024);

  /*
   * allocate/initialize Ugroup / Qgroup
   */

  uint32_t nbOcts = amr_mesh.getNumOctants();
  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";
  
  // uint32_t iGroup = 1;

 // // chose an octant which should have a "same size" neighbor in all direction
  // //uint32_t iOct_local = 2;
  
  // // chose an octant which should have at least
  // // a "larger size" neighbor in one direction
  // //uint32_t iOct_local = 30;

  // // chose an octant which should have at least
  // // an interface with "smaller size" neighbor in one direction
  // uint32_t iOct_local = 26;

  //uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;

  // std::cout << "Looking at octant id = " << iOct_global << "\n";

  // // save solution, just for cross-checking
  // save_solution();

  // std::cout << "Printing U data from iOct = " << iOct_global << "\n";
  // for (uint32_t iz=0; iz<bz; ++iz) {
  //   for (uint32_t iy=0; iy<by; ++iy) {
  //     for (uint32_t ix=0; ix<bx; ++ix) {
  //       uint32_t index = ix + bx*(iy+by*iz);
  //       printf("%5f ", Uhost(index,fm[ID],iOct_global));
  //     }
  //     std::cout << "\n";
  //   }
  //   std::cout << "\n";
  // }

  // compute CFL constraint
  real_t invDt;
  LegacyDataArray Uin(U_);
  ComputeDtHydroFunctor::apply(amr_mesh.getLightOctree(),
                               ComputeDtHydroFunctor::Params(configMap),
                               fm,
                               blockSizes,
                               Uin,
                               invDt);  
  real_t cfl = configMap.getValue<real_t>("hydro", "cfl", 0.5);
  real_t dt = cfl / invDt;

  printf("CFL dt = %f\n",dt);

  // testing MusclBlockGodunovUpdateFunctor
  {
    std::unique_ptr<HydroUpdate> godunov_updater = HydroUpdateFactory::make_instance( "HydroUpdate_hancock",
      configMap,
      foreach_cell,
      timers
    );
    U_.new_fields({"rho_next","e_tot_next","rho_vx_next","rho_vy_next","rho_vz_next"});
    ScalarSimulationData scalar_data;
    scalar_data.set("dt", dt);    
    godunov_updater->update(U_, scalar_data);

    // Kokkos::deep_copy( Uhost, U.U);
    // std::cout << "Printing U data (after update) from iOct = " << iOct_global << "\n";
    // for (uint32_t iy = 0; iy < by; ++iy) {
    //   for (uint32_t ix = 0; ix < bx; ++ix) {
    //     uint32_t index = ix + bx * iy;
    //     printf("%5f ", Uhost(index, fm[ID], iOct_global));
    //   }
    //   std::cout << "\n";
    // }
    // std::cout << "\n";
    
    // DataArrayBlockHost U2_host = Kokkos::create_mirror_view(U2.U);
    // std::cout << "Printing U2 data (after update) from iOct = " << iOct_global << "\n";
    // Kokkos::deep_copy( U2_host, U2.U);
    // for (uint32_t iy = 0; iy < by; ++iy) {
    //   for (uint32_t ix = 0; ix < bx; ++ix) {
    //     uint32_t index = ix + bx * iy;
    //     printf("%5f ", U2_host(index, fm[ID], iOct_global));
    //   }
    //   std::cout << "\n";
    // }
    // std::cout << "\n";
  }

} // run_test



} // namespace dyablo

TEST(dyablo, test_HydroUpdate_hancock_blast_2D)
{
  dyablo::run_test("HydroUpdate_hancock (2D)", 2);
}

TEST(dyablo, test_HydroUpdate_hancock_blast_3D)
{
  dyablo::run_test("HydroUpdate_hancock (3D)", 3);
}

