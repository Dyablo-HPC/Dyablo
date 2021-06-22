#pragma once

#ifdef DYABLO_USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#endif // DYABLO_USE_MPI

#include "kokkos_shared.h"

namespace dyablo {
namespace shared {

class DyabloSession {
private:
  int m_rank, m_nRanks;
#ifdef DYABLO_USE_MPI
  std::unique_ptr<hydroSimu::GlobalMpiSession> mpiSession;
#endif // DYABLO_USE_MPI
public:
  DyabloSession(int argc, char *argv[])
  {
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
#ifdef DYABLO_USE_MPI
    mpiSession = std::make_unique<hydroSimu::GlobalMpiSession>(&argc, &argv);
#endif // DYABLO_USE_MPI

    if (!initialize_kokkos_before_mpi)
      Kokkos::initialize(argc, argv);

    m_rank   = 0;
    m_nRanks = 1;

    {
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

#ifdef DYABLO_USE_MPI
      m_rank   = mpiSession->getRank();
      m_nRanks = mpiSession->getNProc();
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
        std::cout << "I'm MPI task #" << m_rank << " (out of " << m_nRanks
                  << ")"
                  << " pinned to GPU #" << cudaDeviceId << "\n";
      }
#endif // KOKKOS_ENABLE_CUDA
#endif // DYABLO_USE_MPI
    }  // end kokkos config
  }

  ~DyabloSession() { Kokkos::finalize(); }

  int getRank() { return m_rank; }
};

} // namespace shared
} // namespace dyablo