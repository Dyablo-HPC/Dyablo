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

#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/solver_utils.h" // print monitoring information
#include "shared/FieldManager.h"


#include "muscl_block/CopyInnerBlockCellData.h"

using Device = Kokkos::DefaultExecutionSpace;

namespace dyablo { namespace muscl_block {

// =======================================================================
// =======================================================================
void run_test(int argc, char *argv[], uint32_t bSize, uint32_t nbBlocks) {

  /*
   * testing CopyInnerBlockCellDataFunctor
   */
  std::cout << "// ======================================\n";
  std::cout << "Testing CopyInnerBlockCellDataFunctor ...\n";
  std::cout << "// ======================================\n";

  /*
   * read parameter file and initialize a ConfigMap object
   */
  // only MPI rank 0 actually reads input file
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap = broadcast_parameters(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);

  auto fm = fieldMgr.get_id2index();

  // "geometry" setup
  uint32_t ghostWidth = configMap.getInteger("amr", "ghostwidth", 2);

  uint32_t bx = configMap.getInteger("amr", "bx", 0);
  uint32_t by = configMap.getInteger("amr", "by", 0);
  uint32_t bz = configMap.getInteger("amr", "bz", 1);

  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = bx + 2 * ghostWidth;
  uint32_t bz_g = bx + 2 * ghostWidth;

  blockSize_t blockSizes, blockSizes_g;
  blockSizes[IX] = bx;
  blockSizes[IY] = by;
  blockSizes[IZ] = bz;

  blockSizes_g[IX] = bx_g;
  blockSizes_g[IY] = by_g;
  blockSizes_g[IZ] = bz_g;

  uint32_t nbCellsPerOct = params.dimType == TWO_D ? bx * by : bx * by * bz;
  uint32_t nbCellsPerOct_g =
      params.dimType == TWO_D ? bx_g * by_g : bx_g * by_g * bz_g;

  uint32_t nbOctsPerGroup = configMap.getInteger("amr", "nbOctsPerGroup", 32);

  /*
   * allocate/initialize U / Ugroup
   */

  uint32_t nbOcts = 128;

  DataArrayBlock U = DataArrayBlock("U", nbCellsPerOct, params.nbvar, nbOcts);

  DataArrayBlock Ugroup = DataArrayBlock("Ugroup", nbCellsPerOct_g, params.nbvar, nbOctsPerGroup);

  uint32_t iGroup = 1;

  // TODO initialize U, reset Ugroup - write a functor for that

  CopyInnerBlockCellDataFunctor::apply(configMap, params, fm, 
                                       blockSizes, blockSizes_g,
                                       ghostWidth, nbOctsPerGroup,
                                       U, Ugroup, iGroup);

  // TODO - print data from from the chosen iGroup 
  
  // UNFINISHED

} // run_test

} // namespace muscl_block

} // namespace dyablo

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
  dyablo::muscl_block::run_test(argc, argv, bSize, nbBlocks);

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
