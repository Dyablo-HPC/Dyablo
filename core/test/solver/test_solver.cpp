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

#include "shared/DyabloSession.hpp"

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
  shared::DyabloSession mpi_session(argc, argv);

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
  if (mpi_session.getRank()==0) std::cout << "Start computation....\n";
  solver->timers.get("total").start();

  // Hydrodynamics solver time loop
  solver->run();

  // end of computation
  solver->timers.get("total").stop();

  // save last time step
  if (params.nOutput != 0)
    solver->save_solution();
  
  if (mpi_session.getRank()==0) printf("final time is %f\n", solver->m_t);
  
  solver->print_monitoring_info();
  
  delete solver;

  return EXIT_SUCCESS;

} // end main
