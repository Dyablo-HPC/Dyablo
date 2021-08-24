/**
 * \file muscl_convergence.cpp
 *
 * Test MUSCL-Hancock scheme convergence using the isentropic vortex test.
 *
 * MUSCL-Hancock scheme is supposed to be 2nd order.
 * Perform simulation from t=0 to t=1.0 and compute L1 / L2 error.
 *
 * You can increase the range of mesh sizes to test in routine run_test.
 * There is a custom range of each scheme order.
 *
 * For simplicity, this test is assumed to be run with shared memory
 * parallelism only, no MPI.
 *
 * \date July, 5th 2019
 * \author P. Kestener
 */

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "shared/kokkos_shared.h"

#include "shared/FieldManager.h"
#include "shared/HydroParams.h"  // read parameter file
#include "shared/real_type.h"    // choose between single and double precision

#ifdef DYABLO_USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

#include "muscl/SolverHydroMuscl.h"

#include "muscl/init/InitIsentropicVortex.h"
#include "muscl/ComputeError.h"

// banner
//#include "dyablo_version.h"

namespace dyablo {
namespace muscl {

//! data type to store errors values (L1 / L2)
using errors_t = std::array<real_t, 2>;

/**
 * Generate a test parameter file.
 *
 * \param[in] level_min : minimum AMR level
 * \param[in] level_max : maximum AMR level
 *
 */
void generate_input_file(int level_min, int level_max) {

  std::fstream outFile;
  outFile.open("test_isentropic_vortex.ini", std::ios_base::out);

  outFile << "[run]\n";
  outFile << "solver_name=Hydro_Muscl_2D\n";
  outFile << "tEnd=1.0\n";
  outFile << "nStepmax=10000\n";
  outFile << "\n";
  outFile << "# noutput equals -1 means we dump data at every time steps\n";
  outFile << "nOutput=0\n";
  outFile << "\n";
  outFile << "[amr]\n";
  outFile << "level_min=" << level_min << "\n";
  outFile << "level_max=" << level_max << "\n";
  outFile << "\n";
  outFile << "# init step may contain calls to globalRefine\n";
  outFile << "# if the following option is true, we interleave\n";
  outFile << "# calls to load balancing in between globalRefine\n";
  outFile << "# if the following option is false, a single load\n";
  outFile << "# balancing operation is done at the end of\n";
  outFile << "# globalRefine steps\n";
  outFile << "enable_load_balance_during_init=false\n";
  outFile << "\n";
  //outFile << "epsilon_refine=0.01\n";
  //outFile << "epsilon_coarsen=0.002\n";
  outFile << "epsilon_refine=0.005\n";
  outFile << "epsilon_coarsen=0.001\n";
  outFile << "\n";
  outFile << "[mesh]\n";
  outFile << "xmin=0.0\n";
  outFile << "xmax=1.0\n";
  outFile << "\n";
  outFile << "ymin=0.0\n";
  outFile << "ymax=1.0\n";
  outFile << "\n";
  outFile << "# periodic border condition\n";
  outFile << "boundary_type_xmin=3\n";
  outFile << "boundary_type_xmax=3\n";
  outFile << "\n";
  outFile << "boundary_type_ymin=3\n";
  outFile << "boundary_type_ymax=3\n";
  outFile << "\n";
  outFile << "[hydro]\n";
  outFile << "gamma0=1.666\n";
  outFile << "cfl=0.8\n";
  outFile << "niter_riemann=10\n";
  outFile << "iorder=2\n";
  outFile << "slope_type=2\n";
  outFile << "problem=isentropic_vortex\n";
  outFile << "riemann=hllc\n";
  outFile << "\n";
  outFile << "[isentropic_vortex]\n";
  outFile << "density_ambient=1.0\n";
  outFile << "temperature_ambient=1.0\n";
  outFile << "vx_ambient=1.0\n";
  outFile << "vy_ambient=1.0\n";
  outFile << "vz_ambient=1.0\n";
  outFile << "#strength=1.0\n";
  outFile << "scale=0.1\n";
  outFile << "\n";
  outFile << "[output]\n";
  outFile << "outputDir=./\n";
  outFile << "outputPrefix=test_isentropic_vortex_2D\n";
  outFile << "outputVtkAscii=false\n";
  outFile << "write_variables=rho,rho_vx,rho_vy,e_tot\n";
  outFile << "\n";
  outFile << "[other]\n";
  outFile << "implementationVersion=0\n"; 
  outFile << "\n";

  outFile.close();

} // generate_input_file

// ===============================================================
// ===============================================================
// ===============================================================
/**
 * compute both L1 and L2 error between simulation and exact solution.
 *
 * \param[in]  solver is the state of the simulation after a run, 
 *             solution is in solver->U
 *
 * \return an array containing the L1 and L2 errors.
 */
errors_t compute_error_versus_exact(SolverHydroMuscl *solver)
{

  errors_t error;
  
  real_t error_L1 = 0.0;
  real_t error_L2 = 0.0;

  std::shared_ptr<AMRmesh> amr_mesh = solver->amr_mesh;

  int nbvar = solver->params.nbvar;
  int nbCells = amr_mesh->getNumOctants();

  DataArray Uexact = DataArray("Uexact", nbCells, nbvar);

  // retrieve exact solution in auxiliary data arrary : solver.Uaux
  {
    solver->configMap.setBool("isentropic_vortex","use_tEnd",true);

    // retrieve available / allowed names: fieldManager, and field map (fm)
    // necessary to access user data
    auto fm = solver->fieldMgr.get_id2index();


    InitIsentropicVortexDataFunctor::apply(solver->amr_mesh, 
                                           solver->params, 
                                           solver->configMap, 
                                           fm, 
                                           Uexact);
    
    solver->configMap.setBool("isentropic_vortex","use_tEnd",false);
  }
  
  // perform the actual error computation
  {
    error_L1 = 
      Compute_Error_Functor::apply(solver->params,
                                   solver->U,
                                   Uexact,
                                   ID,
                                   NORM_L1);
    error[NORM_L1] = error_L1 / nbCells;
  }
  
  {
    error_L2 = 
      Compute_Error_Functor::apply(solver->params,
                                   solver->U, 
                                   Uexact,
                                   ID,
                                   NORM_L2);
    error[NORM_L2] = sqrt(error_L2) / nbCells;
  }

  return error;

} // compute_error_versus_exact

// ===============================================================
// ===============================================================
// ===============================================================
errors_t test_isentropic_vortex(int level_min, int level_max, real_t tEnd)
{

  using namespace dyablo;

  std::cout << "###############################\n";
  std::cout << "Running isentropic vortex test (MUSCL-Hancock)\n";
  std::cout << "level_min = " << level_min << "\n";
  std::cout << "level_max = " << level_max << "\n";
  std::cout << "###############################\n";

  generate_input_file(level_min, level_max);
  
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = std::string("test_isentropic_vortex.ini");
  ConfigMap configMap(input_file);

  // manually setting tEnd (default value is 10.0)
  if (tEnd > 0)
    configMap.setFloat("run","tEnd",tEnd);
  
  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);
  
  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // initialize workspace memory (U, U2, ...)
  SolverHydroMuscl *solver = (SolverHydroMuscl *) SolverHydroMuscl::create(params, configMap);
  
  if (params.nOutput != 0)
    solver->save_solution();
  
  // start computation
  std::cout << "Start computation....\n";
  solver->timers.get("total").start();

  // Hydrodynamics solver loop
  while ( ! solver->finished() ) {

    solver->next_iteration();

  } // end solver loop

  // end of computation
  solver->timers.get("total").stop();

  // save last time step
  if (params.nOutput != 0)
    solver->save_solution();
  
  printf("final time is %f\n", solver->m_t);

  errors_t error = compute_error_versus_exact(solver);
  
  solver->print_monitoring_info();

  printf("test isentropic vortex for level_min=%d, level_max=%d, error L1=%6.4e, error L2=%6.4e\n",level_min,level_max,error[NORM_L1],error[NORM_L2]);
  
  delete solver;

  return error;
  
} // test_isentropic_vortex

// ===============================================================
// ===============================================================
void run_test()
{
  std::array<real_t, 4> results_L1;
  std::array<real_t, 4> results_L2;

  // setup
  std::array<int, 4> level_min;
  std::array<int, 4> level_max;

  level_min = {4, 5, 6, 7};
  level_max = {4, 5, 6, 7};

  // action
  for (std::size_t i = 0; i<level_min.size(); ++i) {

    int lmin = level_min[i];
    int lmax = level_max[i];

    errors_t error = test_isentropic_vortex(lmin, lmax, -1.0);
    results_L1[i] = error[NORM_L1];
    results_L2[i] = error[NORM_L2];
  }
  
  // report results with norm L1
  for (std::size_t i = 0; i<level_min.size(); ++i) {
    if (i==0)
      printf("level_min=%4d, level_max=%4d, error L1 = %6.4e, order = --   \n",level_min[i],level_max[i],results_L1[i]);
    else
      printf("level_min=%4d, level_max=%4d, error L1 = %6.4e, order = %5.3f\n",level_min[i],level_max[i],results_L1[i],log(results_L1[i-1]/results_L1[i])/log(2.0));
  }
  
  // report results with norm L2
  for (std::size_t i = 0; i<level_min.size(); ++i) {
    if (i==0)
      printf("level_min=%4d, level_max=%4d, error L2 = %6.4e, order = --   \n",level_min[i],level_max[i],results_L2[i]);
    else
      printf("level_min=%4d, level_max=%4d, error L2 = %6.4e, order = %5.3f\n",level_min[i],level_max[i],results_L2[i],log(results_L2[i-1]/results_L2[i])/log(2.0));
  }
  
} // run_test

// ===============================================================
// ===============================================================
void run_test_single(int level_min, int level_max, real_t tEnd)
{
  real_t results_L1;
  real_t results_L2;

  // action
  errors_t error = test_isentropic_vortex(level_min, level_max, tEnd);
  results_L1 = error[NORM_L1];
  results_L2 = error[NORM_L2];
  
  // report results with norm L1
  printf("level_min=%4d, level_max=%4d, error L1 = %6.4e, order = --   \n",level_min,level_max,results_L1);
  
  // report results with norm L2
  printf("level_min=%4d, level_max=%4d, error L2 = %6.4e, order = --   \n",level_min,level_max,results_L2);
  
} // run_test_single

} // namespace muscl

} // namespace dyablo

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char *argv[]) {

  using namespace dyablo::muscl;

  // Create MPI session if MPI enabled
#ifdef DYABLO_USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc, &argv);
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
    if (Kokkos::hwloc::available()) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE[" << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT[" << Kokkos::hwloc::get_available_threads_per_core()
          << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";

#ifdef DYABLO_USE_MPI
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
#endif // DYABLO_USE_MPI
  }

  // banner
  // if (rank==0) print_version_info();

  // check command line for another order to test
  if (argc == 1) {

    run_test();

  } else if (argc == 2) {

    int level_min = std::atoi(argv[1]);
    int level_max = level_min;
    real_t tEnd = -1.0;

    run_test_single(level_min, level_max, tEnd);

  } else if (argc>=3) {

    int level_min = std::atoi(argv[1]);
    int level_max = std::atoi(argv[2]);
    real_t tEnd = -1.0;
    if (argc>3)
      tEnd = std::atof(argv[3]);

    run_test_single(level_min, level_max, tEnd);

  }

  Kokkos::finalize();

  return EXIT_SUCCESS;

} // end main
