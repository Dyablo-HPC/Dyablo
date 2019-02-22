#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_PABLO.hpp"

using namespace std;
using namespace bitpit;

// ======================================================================== //
/**
 *  \example PABLO_test0.cpp
 */
// ======================================================================== //

/**
 * Run the example.
 */
void run()
{

  //int nProcs;
  int rank;
#if BITPIT_ENABLE_MPI==1
  //MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  //nProcs = 1;
  rank   = 0;
#endif

  /**<Instantation of a 2D pablo uniform object.*/
  PabloUniform amr_mesh(2);

  amr_mesh.adaptGlobalRefine();

  for (int iter=0; iter<3; ++iter) {

    // refine all cells
    amr_mesh.adaptGlobalRefine();
    //uint32_t nocts = amr_mesh.getNumOctants();
    
    amr_mesh.setMarker(3,1);
    amr_mesh.setMarker(8,1);
    amr_mesh.adapt();

#if BITPIT_ENABLE_MPI==1
    /**<(Load)Balance the octree over the processes.*/
    amr_mesh.loadBalance();
#endif

    amr_mesh.updateConnectivity();  
    
    /**<Define vectors of data.*/
    uint32_t nocts2 = amr_mesh.getNumOctants();
    vector<double> oct_data(nocts2, 0.0);
    
    /**<Assign a data to the octants with at least one node inside the circle.*/
    for (unsigned int i=0; i<nocts2; i++){
      oct_data[i] = rank;
    }
    
    amr_mesh.updateConnectivity();
    amr_mesh.writeTest("PABLO_test0_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data);

  } // end for iter
} // end run

/*!
 * Main program.
 */
int main(int argc, char *argv[])
{
#if BITPIT_ENABLE_MPI==1
  MPI_Init(&argc,&argv);
#else
  BITPIT_UNUSED(argc);
  BITPIT_UNUSED(argv);
#endif

  int nProcs;
  int rank;
#if BITPIT_ENABLE_MPI==1
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  nProcs = 1;
  rank   = 0;
#endif

  // Initialize the logger
  log::manager().initialize(log::SEPARATE, false, nProcs, rank);
  log::cout() << fileVerbosity(log::NORMAL);
  log::cout() << consoleVerbosity(log::QUIET);

  // Run the example
  try {
    run();
  } catch (const std::exception &exception) {
    log::cout() << exception.what();
    exit(1);
  }

#if BITPIT_ENABLE_MPI==1
  MPI_Finalize();
#endif
}
