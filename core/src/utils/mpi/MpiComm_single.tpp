#include <cstdint>
#include <algorithm>
#include <cassert>

namespace dyablo{

//! Construct a MpiComm for a given MPI communicator
MpiComm::MpiComm(MPI_Comm_t mpi_comm_id)
  : mpi_comm_id(mpi_comm_id), mpi_comm_size(1), mpi_comm_rank(0)
{}

MpiComm::MpiComm()
  : MpiComm(0)
{}

void MpiComm::MPI_Barrier()
{ /* empty */ }

template<typename T>
void MpiComm::MPI_Allreduce( const T* sendbuf, T* recvbuf, int count, MPI_Op_t op ) const
{
  std::copy( sendbuf, sendbuf+count-1, recvbuf );
}

template<typename T>
void MpiComm::MPI_Bcast( T* buffer, int count, int root ) const
{ /* empty */ }

template<typename T>
void MpiComm::MPI_Alltoall( const T* sendbuf, int sendcount, T* recvbuf, int recvcount ) const
{
  std::copy( sendbuf, sendbuf+sendcount-1, recvbuf );
}

template<typename Kokkos_View_t>
MpiComm::MPI_Request_t MpiComm::MPI_Isend( const Kokkos_View_t& view, int dest, int tag ) const
{
  DYABLO_ASSERT_HOST_RELEASE( false, "Cannot Isend in MpiComm single mode" );
  return 0;
}

template<typename Kokkos_View_t>
MpiComm::MPI_Request_t MpiComm::MPI_Irecv( const Kokkos_View_t& view, int dest, int tag ) const
{
  DYABLO_ASSERT_HOST_RELEASE( false, "Cannot Irecv in MpiComm single mode" );
  return 0;
}

template<typename T>
void MpiComm::MPI_Send( const T* buffer, int count, int dest, int tag ) const
{
  DYABLO_ASSERT_HOST_RELEASE( false, "Cannot MPI_Send in MpiComm single mode" );
}

template<typename T>
void MpiComm::MPI_Recv( T* view, int count, int dest, int tag ) const
{
  DYABLO_ASSERT_HOST_RELEASE( false, "Cannot MPI_Recv in MpiComm single mode" );
}

inline void MpiComm::MPI_Waitall( int count, MPI_Request_t* array_of_requests ) const
{ /*nothing to do*/ }

} // namespace dyablo