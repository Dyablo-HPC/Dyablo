/**
 * \file test_AMRMetaData.cpp
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

#include "shared/AMRMetaData.h"

#include <iostream>

using Device = Kokkos::DefaultExecutionSpace;


namespace dyablo
{

// ==========================================================
// ==========================================================
void run_test(int dim)
{

  std::cout << "==================================\n";
  std::cout << "AMRMetaData test in dim : " << dim << "\n";
  std::cout << "==================================\n";

  // stage 1 : create a PABLO mesh

  int niter = 3;

  /**<Instantation of a nDimensional pablo uniform object.*/
  bitpit::PabloUniform amr_mesh(dim);

  /**<set periodic border condition */
  amr_mesh.setPeriodic(0);
  amr_mesh.setPeriodic(1);
  amr_mesh.setPeriodic(2);
  amr_mesh.setPeriodic(3);
  amr_mesh.setPeriodic(4);
  amr_mesh.setPeriodic(5);


  /**<Refine globally four level and write the octree.*/
  for (int iter=1; iter<niter; ++iter)
  {
    
    amr_mesh.adaptGlobalRefine();

    amr_mesh.setMarker(3,1);
    amr_mesh.setMarker(5,1);
    //amr_mesh.setMarker(8,1);
    amr_mesh.adapt();
    
#if BITPIT_ENABLE_MPI==1
    /**<(Load)Balance the octree over the processes.*/
    amr_mesh.loadBalance();
#endif

    amr_mesh.updateConnectivity();

  }  
  amr_mesh.updateConnectivity();

  std::cout << "Number of octants :" << amr_mesh.getNumOctants() <<  "\n";

  // stage 2 : create a AMRMetaData ovject

  uint64_t capacity = 1024*1024;
  AMRMetaData amrMetadata(capacity);

  amrMetadata.report();
  amrMetadata.update(amr_mesh);
  amrMetadata.report();

  {

    // device map
    auto map_device = amrMetadata.hashmap();

    // host mirror
    AMRMetaData::hashmap_t::HostMirror map_host(map_device.capacity());

    // copy on host before printing
    Kokkos::deep_copy(map_host, map_device);

    std::cout << "Print hashmap\n";
    for (std::size_t i=0; i<map_host.capacity(); ++i)
    {
      if (map_host.valid_at(i)) 
      {
        auto key   = map_host.key_at(i)[0];
        auto level = map_host.key_at(i)[1];
        auto value = map_host.value_at(i);

        std::cout << i << " "
                  << "map[" << key << "]=" << value
                  << " (level=" << level << ")"
                  << " and Morton (from Pablo) = " << amr_mesh.getMorton(value)
                  << "\n";
      }
    }

  }

} // run_test

} // dyablo


// ==========================================================
// ==========================================================
int main(int argc, char* argv[])
{
  
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

  // dim 2
  dyablo::run_test(2);

  // dim3
  dyablo::run_test(3);

  Kokkos::finalize();

  return EXIT_SUCCESS;


} // main
