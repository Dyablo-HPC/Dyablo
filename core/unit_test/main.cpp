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

#ifdef DYABLO_USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

struct ExecutionEnvironmentScopeGuard
{
  ExecutionEnvironmentScopeGuard(int argc, char *argv[])
  {
    Kokkos::initialize(argc, argv);

    int rank = 0;
    int nRanks = 1;

    {
      std::cout << "##########################\n";
      std::cout << "KOKKOS CONFIG             \n";
      std::cout << "##########################\n";
      
      std::ostringstream msg;
      std::cout << "Kokkos configuration" << std::endl;
      if ( Kokkos::hwloc::available() ) {
        msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
            << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
            << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
            << "] )"
            << std::endl ;
      }
      Kokkos::print_configuration( msg );
      std::cout << msg.str();
      std::cout << "##########################\n";
      
#ifdef DYABLO_USE_MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
# ifdef KOKKOS_ENABLE_CUDA
      {
        
        // To enable kokkos accessing multiple GPUs don't forget to
        // add option "--ndevices=X" where X is the number of GPUs
        // you want to use per node.
        
        // on a large cluster, the scheduler should assign ressources
        // in a way that each MPI task is mapped to a different GPU
        // let's cross-checked that:
        
        int cudaDeviceId;
        cudaGetDevice(&cudaDeviceId);
        std::cout << "I'm MPI task #" << rank << " (out of " << nRanks << ")"
                  << " pinned to GPU #" << cudaDeviceId << "\n";
        
      }
# endif // KOKKOS_ENABLE_CUDA
#endif // DYABLO_USE_MPI
    }    // end kokkos config
    
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
#ifdef DYABLO_USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // DYABLO_USE_MPI
  
  ExecutionEnvironmentScopeGuard scope_guard(argc, argv);
  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
