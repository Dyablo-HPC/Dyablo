#include <cstdint>
#include <algorithm>

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
  std::copy( sendbuf, sendbuf+count, recvbuf );
}

template<typename T>
void MpiComm::MPI_Bcast( T* buffer, int count, int root ) const
{ /* empty */ }

} // namespace dyablo