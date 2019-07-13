/**
 * \file test_hdf5.cpp
 * \author Pierre Kestener
 */

#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include <string>

#include "bitpit_PABLO.hpp"

#include "utils/config/ConfigMap.h"
#include "shared/HydroParams.h"
#include "shared/HDF5_IO.h"

using namespace bitpit;

/**
 * Run the example.
 */
void run(std::string input_filename)
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

  /// initialize configMap and HydroParameter
  ConfigMap configMap = broadcast_parameters(input_filename);
  HydroParams params = HydroParams();
  params.setup(configMap);

  /// Instantation of a 2D pablo uniform object.
  std::shared_ptr<dyablo::AMRmesh> amr_mesh = std::make_shared<dyablo::AMRmesh>(2);

  amr_mesh->adaptGlobalRefine();

  // create HDF5 writer
  dyablo::HDF5_Writer writer(amr_mesh, configMap, params);

  for (int iter=0; iter<3; ++iter) {

    // refine all cells
    amr_mesh->adaptGlobalRefine();
    //uint32_t nocts = amr_mesh->getNumOctants();
    
    amr_mesh->setMarker(3,1);
    amr_mesh->setMarker(8,1);
    amr_mesh->adapt();

#if BITPIT_ENABLE_MPI==1
    /**<(Load)Balance the octree over the processes.*/
    amr_mesh->loadBalance();
#endif

    amr_mesh->updateConnectivity();  
    
    /**<Define vectors of data.*/
    uint32_t nocts2 = amr_mesh->getNumOctants();
    vector<double> oct_data(nocts2, 0.0);
    
    /**<Assign a data to the octants with at least one node inside the circle.*/
    for (unsigned int i=0; i<nocts2; i++){
      oct_data[i] = rank;
    }
    
    amr_mesh->updateConnectivity();
    //amr_mesh->writeTest("PABLO_test0_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data);

    // save hdf5 data
    {
      hid_t output_type = H5T_NATIVE_DOUBLE;

      // // open the new file and write our stuff
      // writer.open("test_hdf5");
      // writer.write_header(0.0);

      // // close the file
      // writer.write_footer();
      // writer.close();

    }

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
    if (argc>1) {
      std::string input_file = std::string(argv[1]);
      run(input_file);
    } else {
      std::cerr << "argc must be larger than 1. Please provide ini input parameter file.\n";
    }
  } catch (const std::exception &exception) {
    log::cout() << exception.what();
    exit(1);
  }

#if BITPIT_ENABLE_MPI==1
  MPI_Finalize();
#endif
}
