#pragma once

#ifdef DYABLO_USE_MPI
#include <mpi.h>
#endif

namespace dyablo{

/// \brief Object representation of an MPI communicator.
class MpiComm
{
public:
  #ifdef DYABLO_USE_MPI
  using MPI_Comm_t = MPI_Comm;
  #else
  using MPI_Comm_t = int;
  #endif
  
  enum MPI_Op_t{
    MIN,
    MAX,
    SUM,
    NUM_OPS
  };
private:
  /// Construct a MpiComm for a given MPI communicator
  inline MpiComm(MPI_Comm_t mpi_comm_id);
  /// Construct MpiComm from MPI_COMM_WORLD
  inline MpiComm();
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

#ifdef DYABLO_USE_MPI
#include "MpiComm_mpi.tpp"
#else
#include "MpiComm_single.tpp"
#endif