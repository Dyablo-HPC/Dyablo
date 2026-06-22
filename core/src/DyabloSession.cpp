#include "DyabloSession.hpp"

#include <iostream>

#include <Kokkos_Core.hpp>

#include "utils/misc/Dyablo_assert.h"
#include "utils/mpi/GlobalMpiSession.h"
#include "utils/mpi/MpiBufferPool.h"

#include "dyablo_version.h"

namespace dyablo {

DyabloSession*& DyabloSession::unique_session()
{
  static DyabloSession* session_ptr = nullptr;
  return session_ptr;
}

[[maybe_unused]] static bool MPI_enabled()
{
  #ifdef DYABLO_USE_MPI
  return true;
  #else // DYABLO_USE_MPI
  return false;
  #endif // DYABLO_USE_MPI
}

DyabloSession::DyabloSession(int& argc, char *argv[])
{
  DYABLO_ASSERT_HOST_RELEASE( !unique_session(), "Dyablo Session already running!" );

  using namespace dyablo;

  bool initialize_kokkos_before_mpi = false;

#ifdef KOKKOS_ENABLE_CUDA
  if (std::getenv("PSM2_CUDA") != NULL)
  {
    std::cout << "PSM2_CUDA detected : Initializing Kokkos before MPI" << std::endl;
    initialize_kokkos_before_mpi = true;
  }
#endif

  if (initialize_kokkos_before_mpi)
    Kokkos::initialize(argc, argv);

    // Create MPI session if MPI enabled
  mpiSession = std::make_unique<GlobalMpiSession>(&argc, &argv);
  if (!initialize_kokkos_before_mpi)
    Kokkos::initialize(argc, argv);

  int rank   = GlobalMpiSession::get_comm_world().MPI_Comm_rank();

  if( rank == 0 )
  {
    print_version_info();

    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available())
    {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE[" << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT[" << Kokkos::hwloc::get_available_threads_per_core()
          << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";
  }

#ifdef KOKKOS_ENABLE_CUDA
  if( MPI_enabled() )
  {

    // To enable kokkos accessing multiple GPUs don't forget to
    // add option "--ndevices=X" where X is the number of GPUs
    // you want to use per node.

    // on a large cluster, the scheduler should assign ressources
    // in a way that each MPI task is mapped to a different GPU
    // let's cross-checked that:

    int nRanks = GlobalMpiSession::get_comm_world().MPI_Comm_size();

    int cudaDeviceId;
    cudaGetDevice(&cudaDeviceId);
    std::cout << "I'm MPI task #" << rank << " (out of " << nRanks
              << ")"
              << " pinned to GPU #" << cudaDeviceId << "\n";
  }
#endif // KOKKOS_ENABLE_CUDA

  this->mpi_pool = std::make_unique<MpiBufferPool>();

  unique_session() = this;
}

DyabloSession::~DyabloSession() 
{ 
  for( auto f : finalize_callbacks )
  {
    f();
  }
  this->mpi_pool.reset();
  Kokkos::finalize(); 
  unique_session() = nullptr;
}

DyabloSession& DyabloSession::get_DyabloSession()
{
  DYABLO_ASSERT_HOST_RELEASE( unique_session(), "No dyablo session running!" );

  return *(unique_session());
}

MpiBufferPool& DyabloSession::get_MpiBufferPool()
{
  return *(get_DyabloSession().mpi_pool);
}
  
void DyabloSession::call_before_finalize( std::function<void()> finalize_callback )
{
  std::function<void()> finalize_callback_std(finalize_callback);
  finalize_callbacks.push_back( finalize_callback_std );
}

} // namespace dyablo
