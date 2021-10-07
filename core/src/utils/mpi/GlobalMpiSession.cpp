#include "GlobalMpiSession.h"

#ifdef DYABLO_USE_MPI
#include <mpi.h>
#endif

namespace dyablo {

GlobalMpiSession::GlobalMpiSession( int* argc, char*** argv )
{
  #ifdef DYABLO_USE_MPI
  MPI_Init(argc, argv);
  #endif
}
  
GlobalMpiSession::~GlobalMpiSession()
{
  #ifdef DYABLO_USE_MPI
  MPI_Finalize();
  #endif
}

} // namespace dyablo
