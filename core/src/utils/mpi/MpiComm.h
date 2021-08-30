#pragma once

#include <mpi.h>

namespace dyablo{

/// \brief Object representation of an MPI communicator.
class MpiComm
{
public:
  using MPI_Comm_t = MPI_Comm;
  enum MPI_Op_t{
    MIN,
    MAX,
    SUM,
    NUM_OPS
  };
private:
  /// Construct a MpiComm for a given MPI communicator
  inline MpiComm(MPI_Comm_t mpi_comm_id = MPI_COMM_WORLD);
public:
  /// Get an object representing MPI_COMM_WORLD 
  inline static MpiComm& world()
  {
    static MpiComm res;
    return res;
  }

  inline MPI_Comm_t getId() const
  {return mpi_comm_id;}

  inline int MPI_Comm_rank() const
  { return mpi_comm_rank; }
  inline int MPI_Comm_size() const
  { return mpi_comm_size; }

  inline void MPI_Barrier();

  template<typename T>
  void MPI_Allreduce( const T* sendbuf, T* recvbuf, int count, MPI_Op_t op ) const;

  template<typename T>
  void MPI_Bcast( T* buffer, int count, int root ) const;

private:
  MPI_Comm_t mpi_comm_id;
  int mpi_comm_size, mpi_comm_rank;
};

} // namespace dyablo

#include "MpiComm_mpi.tpp"