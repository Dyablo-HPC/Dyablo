/**
 * dyablo test solver.
 *
 * \date January, 20th 2019
 * \author P. Kestener
 */

#include <cstdlib>
#include <cstdio>
#include <string>

#include "shared/kokkos_shared.h"

#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/solver_utils.h" // print monitoring information
#include "shared/FieldManager.h"

#ifdef DYABLO_USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

#include "muscl/SolverHydroMuscl.h"
#include "muscl_block/SolverHydroMusclBlock.h"

// banner
//#include "dyablo_version.h"

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char *argv[])
{

  using namespace dyablo;

  // Create MPI session if MPI enabled
#ifdef DYABLO_USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // DYABLO_USE_MPI
  
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

  // banner
  //if (rank==0) print_version_info();

  /*
   * read parameter file and initialize a ConfigMap object
   */
  // only MPI rank 0 actually reads input file
  std::string input_file = std::string(argv[1]);
  ConfigMap configMap = broadcast_parameters(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // initialize workspace memory (U, U2, ...)
  SolverBase *solver;
  if (solver_name.find("Muscl_Block") != std::string::npos) {
    solver = muscl_block::SolverHydroMusclBlock::create(params, configMap);
  } else {
    solver = muscl::SolverHydroMuscl::create(params, configMap);
  }

  // start computation
  if (rank==0) std::cout << "Start computation....\n";
  solver->m_timers[TIMER_TOTAL]->start();

  // Hydrodynamics solver time loop
  solver->run();

  // end of computation
  solver->m_timers[TIMER_TOTAL]->stop();

  // save last time step
  if (params.nOutput != 0)
    solver->save_solution();
  
  if (rank==0) printf("final time is %f\n", solver->m_t);
  
  solver->print_monitoring_info();
  
  delete solver;

  Kokkos::finalize();
  
  return EXIT_SUCCESS;

} // end main
