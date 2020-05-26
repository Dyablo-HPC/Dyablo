/**
 * @file main.cpp
 * @date May 25, 2020
 * @author pkestene
 * @note
 *
 * Contributors: pkestene
 *
 * This file is part of the dyablo software project.
 *
 * @copyright © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
 *
 */

#include <Kokkos_Core.hpp>

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

struct ExecutionEnvironmentScopeGuard
{
  ExecutionEnvironmentScopeGuard(int argc, char *argv[])
  {
    Kokkos::initialize(argc, argv);
  }
  ~ExecutionEnvironmentScopeGuard()
  {
    Kokkos::finalize();
  }
};

// initialization function
bool init_function() { return true; }

// entry point
int main(int argc, char *argv[])
{
  // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI
  
  ExecutionEnvironmentScopeGuard scope_guard(argc, argv);
  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
