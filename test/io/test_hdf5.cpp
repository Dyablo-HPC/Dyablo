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
#include "shared/FieldManager.h"
#include "shared/HDF5_IO.h"
#include "shared/io_utils.h"

// don't you dare calling "using namespace" ever again !
//using namespace bitpit;

/*
 * This test is meant to run entirely on CPU (not GPU)
 * so we define and enforce a data array type on host memory space
 */
using DataArray     = dyablo::DataArray;
using DataArrayHost = dyablo::DataArrayHost;

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

  // should we write multiple cell per octree leaf ?
  bool write_block_data = configMap.getBool("amr", "use_block_data", false);
  int bx = configMap.getInteger("amr", "bx", 0);
  int by = configMap.getInteger("amr", "by", 0);
  int bz = configMap.getInteger("amr", "bz", 0);

  int nbCellsPerLeaf = 1;

  if (write_block_data) {
    nbCellsPerLeaf = params.dimType == TWO_D ? 
      bx * by : 
      bx * by * bz;
  }

  // variable map
  str2int_t names2index; // this is initially empty
  dyablo::build_var_to_write_map(names2index, params, configMap);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);
  auto fm = fieldMgr.get_id2index();


  /// Instantation of a 2D pablo uniform object.
  std::shared_ptr<dyablo::AMRmesh> amr_mesh = std::make_shared<dyablo::AMRmesh>(2);

  amr_mesh->adaptGlobalRefine();

  // create HDF5 writer
  dyablo::HDF5_Writer writer(amr_mesh, configMap, params);

  for (int iter=0; iter<2; ++iter) {

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
    std::vector<double> oct_data(nocts2, 0.0);
    
    /**<Assign a data to the octants with at least one node inside the circle.*/
    for (unsigned int i=0; i<nocts2; i++){
      oct_data[i] = rank;
    }

    amr_mesh->updateConnectivity();
    //amr_mesh->writeTest("PABLO_test0_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data);

    printf("local  num octants : %ld\n",amr_mesh->getNumOctants());
    printf("global num octants : %ld\n",amr_mesh->getGlobalNumOctants());

    // create some fake data - 2 scalar value but only one initialized
    // the idea for this test is allocated at least 2 variables
    // to have actual left/right layout
    // out Hdf5 writer does something different whether data has
    // left or right layout
    DataArray userdata = DataArray("fake_data",amr_mesh->getNumOctants()*nbCellsPerLeaf,2);
    DataArrayHost userdatah = Kokkos::create_mirror(userdata);

    // initialize on host (to enable the use of PABLO amr_mesh)
    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::OpenMP>(0, amr_mesh->getNumOctants()),
      [=](int i) {
        for (int j = 0; j < nbCellsPerLeaf; ++j) {
          userdatah(i * nbCellsPerLeaf + j, fm[ID]) =
            amr_mesh->getGlobalIdx((uint32_t)0) + i;
          userdatah(i * nbCellsPerLeaf + j, fm[IP]) =
            (amr_mesh->getGlobalIdx((uint32_t)0) + i) * nbCellsPerLeaf + j;
        }
      });

    //Kokkos::OpenMP().fence();
    //Kokkos::deep_copy(userdata, userdatah);

    // save hdf5 data
    {
      hid_t output_type = H5T_NATIVE_DOUBLE;

      writer.update_mesh_info();

      // open the new file and write our stuff
      std::string prefix = configMap.getString("output", "outputPrefix", "output");
      writer.open(prefix+"_iter"+std::to_string(iter), "./");
      writer.write_header(1.0*iter);

      // write user the fake data (all scalar fields, here only one)
      writer.write_quadrant_attribute(userdatah, fm, names2index);

      // close the file
      writer.write_footer();
      writer.close();

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

  // initialize kokkos 
  Kokkos::initialize(argc, argv);
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
  }

  // Initialize the logger
  bitpit::log::manager().initialize(bitpit::log::SEPARATE, false, nProcs, rank);
  bitpit::log::cout() << fileVerbosity(bitpit::log::NORMAL);
  bitpit::log::cout() << consoleVerbosity(bitpit::log::QUIET);

  // Run the example
  try {
    if (argc>1) {
      std::string input_file = std::string(argv[1]);
      run(input_file);
    } else {
      std::cerr << "argc must be larger than 1. Please provide ini input parameter file.\n";
    }
  } catch (const std::exception &exception) {
    bitpit::log::cout() << exception.what();
    exit(1);
  }

  Kokkos::finalize();

#if BITPIT_ENABLE_MPI==1
  MPI_Finalize();
#endif
}
