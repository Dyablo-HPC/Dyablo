#include <mpi.h>

#include <cstdint>

namespace dyablo
{

namespace MpiComm_impl{
  template< typename T > constexpr MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
  template<> constexpr MPI_Datatype mpi_type<double>    = MPI_DOUBLE;
  template<> constexpr MPI_Datatype mpi_type<float>     = MPI_FLOAT;
  template<> constexpr MPI_Datatype mpi_type<uint64_t>  = MPI_UINT64_T;
  template<> constexpr MPI_Datatype mpi_type<uint32_t>  = MPI_UINT32_T;
  template<> constexpr MPI_Datatype mpi_type<uint16_t>  = MPI_UINT16_T;
  template<> constexpr MPI_Datatype mpi_type<uint8_t>   = MPI_UINT8_T;
  template<> constexpr MPI_Datatype mpi_type<int64_t>   = MPI_INT64_T;
  template<> constexpr MPI_Datatype mpi_type<int32_t>   = MPI_INT32_T;
  template<> constexpr MPI_Datatype mpi_type<int16_t>   = MPI_INT16_T;
  template<> constexpr MPI_Datatype mpi_type<int8_t>    = MPI_INT8_T;
  template<> constexpr MPI_Datatype mpi_type<char>      = MPI_CHAR;

  constexpr std::array<MPI_Op, MpiComm::MPI_Op_t::NUM_OPS> mpi_op{
    MPI_MIN,
    MPI_MAX,
    MPI_SUM
  };

} // namespace MpiComm_impl

//! Construct a MpiComm for a given MPI communicator
MpiComm::MpiComm(MPI_Comm_t mpi_comm_id)
  : mpi_comm_id(mpi_comm_id)
{
  ::MPI_Comm_size( mpi_comm_id, &this->mpi_comm_size );
  ::MPI_Comm_rank( mpi_comm_id, &this->mpi_comm_rank );
}

void MpiComm::MPI_Barrier()
{
  ::MPI_Barrier( mpi_comm_id );
}

template<typename T>
void MpiComm::MPI_Allreduce( const T* sendbuf, T* recvbuf, int count, MPI_Op_t op ) const
{
  using namespace MpiComm_impl;
  ::MPI_Allreduce( sendbuf, recvbuf, count, mpi_type<T>, mpi_op[op], mpi_comm_id);
}

template<typename T>
void MpiComm::MPI_Bcast( T* buffer, int count, int root ) const
{
  using namespace MpiComm_impl;
  ::MPI_Bcast( buffer, count, mpi_type<T>, root, mpi_comm_id);
}

} // namespace dyablo