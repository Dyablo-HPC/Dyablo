/**
 * \file test_CopyInnerBlockCellData.cpp
 * \author Pierre Kestener
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

#include "muscl_block/CopyInnerBlockCellData.h"

using Device = Kokkos::DefaultExecutionSpace;

// =======================================================================
// =======================================================================
void run_test(uint32_t bSize, uint32_t nbBlocks) {

  /*
   * testing CopyInnerBlockCellDataFunctor
   */
  std::cout << "// ======================================\n";
  std::cout << "Testing CopyInnerBlockCellDataFunctor ...\n";
  std::cout << "// ======================================\n";
  
  // TODO
  
} // run_test

// =======================================================================
// =======================================================================
int main(int argc, char *argv[]) {

  // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc, &argv);
#endif // USE_MPI

  Kokkos::initialize(argc, argv);

  int rank = 0;
  int nRanks = 1;

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available()) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE[" << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT[" << Kokkos::hwloc::get_available_threads_per_core()
          << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#ifdef KOKKOS_ENABLE_CUDA
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
#endif // KOKKOS_ENABLE_CUDA
#endif // USE_MPI
  }    // end kokkos config

  uint32_t bSize = 4;
  uint32_t nbBlocks = 32;
  run_test(bSize, nbBlocks);

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
