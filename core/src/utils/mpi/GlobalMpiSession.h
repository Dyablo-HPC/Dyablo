/**
 * \brief A wrapper class around MPI_Init and MPI_Finalize 
 *
 * 
 * Official MPI C++ API is deprecated : this is an attempt at making a minimal 
 * MPI C++ interface for Dyablo. This uses a modified version of MpiComm.h from trilinos 
 *
 * \author Arnaud Durocher
 * \date 08/2021
 * 
 * See copyright of the original version.
 */
#pragma once

#include "MpiComm.h"

namespace dyablo {

/** \brief This class provides methods for initializing, finalizing, and
 * querying the global MPI session.
 *
 * This class is primarilly designed to insulate basic <tt>main()</tt>
 * program type of code from having to know if MPI is enabled or not.
 */
class GlobalMpiSession
{
public: 
  /// \brief Calls MPI_Init() if DYABLO_USE_MPI is enabled.
  GlobalMpiSession( int* argc, char*** argv );
  
  /// \brief Calls MPI_Finalize() if DYABLO_USE_MPI is enabled.
  ~GlobalMpiSession();

  /// Get MPI_COMM_WORLD communicator
  static MpiComm& get_comm_world()
  {
    return MpiComm::world();
  }
};

} // namespace dyablo
