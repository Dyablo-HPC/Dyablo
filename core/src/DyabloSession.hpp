#pragma once

#include <iostream>
#include <memory>
#include "utils/mpi/GlobalMpiSession.h"

#include "kokkos_shared.h"

namespace dyablo {


class DyabloSession {
private:
  int m_rank, m_nRanks;
  std::unique_ptr<dyablo::GlobalMpiSession> mpiSession;
public:
  DyabloSession(int& argc, char *argv[])
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
    mpiSession = std::make_unique<GlobalMpiSession>(&argc, &argv);
    if (!initialize_kokkos_before_mpi)
      Kokkos::initialize(argc, argv);

    m_rank   = GlobalMpiSession::get_comm_world().MPI_Comm_rank();
    m_nRanks = GlobalMpiSession::get_comm_world().MPI_Comm_size();

    if( m_rank == 0 )
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

      int cudaDeviceId;
      cudaGetDevice(&cudaDeviceId);
      std::cout << "I'm MPI task #" << m_rank << " (out of " << m_nRanks
                << ")"
                << " pinned to GPU #" << cudaDeviceId << "\n";
    }
#endif // KOKKOS_ENABLE_CUDA
  }

  ~DyabloSession() { Kokkos::finalize(); }

  int getRank() { return m_rank; }

  static bool MPI_enabled()
  {
    #ifdef DYABLO_USE_MPI
    return true;
    #else // DYABLO_USE_MPI
    return false;
    #endif // DYABLO_USE_MPI
  }
};


} // namespace dyablo
