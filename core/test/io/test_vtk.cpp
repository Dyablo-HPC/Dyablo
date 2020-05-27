/**
 * \file test_vtk.cpp
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
#include "shared/SimpleVTKIO.h"
#include "shared/io_utils.h"

//using namespace bitpit;

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

  // 2:1 balance
  int codim = configMap.getInteger("amr", "codim", amr_mesh->getDim());
  amr_mesh->setBalanceCodimension(codim);

  uint32_t idx = 0;
  amr_mesh->setBalance(idx,true);

  amr_mesh->adaptGlobalRefine();

  for (int iter=0; iter<3; ++iter) {

    // refine all cells
    amr_mesh->adaptGlobalRefine();
    //uint32_t nocts = amr_mesh->getNumOctants();
    
    amr_mesh->setMarker(3,1);
    amr_mesh->setMarker(5,1);
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
    DataArray userdata = DataArray("fake_data",amr_mesh->getNumOctants(),2);
    DataArrayHost userdatah = Kokkos::create_mirror(userdata);
    
    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::OpenMP>(0, amr_mesh->getNumOctants()), 
      [=] (int i) {
        userdatah(i,fm[ID])=amr_mesh->getGlobalIdx((uint32_t) 0)+i;
      });

    Kokkos::OpenMP().fence();
    Kokkos::deep_copy(userdata, userdatah);

    // save vtk data
    {

      std::string prefix = configMap.getString("output", "outputPrefix", "output");

      dyablo::writeVTK(*amr_mesh, prefix+"_iter"+std::to_string(iter), userdata, fm, names2index, configMap);

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