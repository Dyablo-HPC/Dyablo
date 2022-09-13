#include <mpi.h>

#include <cstdint>
#include <array>
#include <vector>

namespace dyablo
{

namespace MpiComm_impl{
  template< typename T > inline MPI_Datatype mpi_type(){ 
    static_assert( !std::is_same<T,T>::value, "Unknown MPI type");
    return MPI_DATATYPE_NULL;
  }
  template<> inline MPI_Datatype mpi_type<double>(){return MPI_DOUBLE;}
  template<> inline MPI_Datatype mpi_type<float>(){return MPI_FLOAT;}
  template<> inline MPI_Datatype mpi_type<uint64_t>(){return MPI_UINT64_T;}
  template<> inline MPI_Datatype mpi_type<uint32_t>(){return MPI_UINT32_T;}
  template<> inline MPI_Datatype mpi_type<uint16_t>(){return MPI_UINT16_T;}
  template<> inline MPI_Datatype mpi_type<uint8_t>(){return MPI_UINT8_T;}
  template<> inline MPI_Datatype mpi_type<int64_t>(){return MPI_INT64_T;}
  template<> inline MPI_Datatype mpi_type<int32_t>(){return MPI_INT32_T;}
  template<> inline MPI_Datatype mpi_type<int16_t>(){return MPI_INT16_T;}
  template<> inline MPI_Datatype mpi_type<int8_t>(){return MPI_INT8_T;}
  template<> inline MPI_Datatype mpi_type<char>(){return MPI_CHAR;}
  template<> inline MPI_Datatype mpi_type<bool>(){return MPI_CXX_BOOL;}

  const std::array<MPI_Op, MpiComm::MPI_Op_t::NUM_OPS> mpi_op{
    MPI_MIN,
    MPI_MAX,
    MPI_SUM,
    MPI_LOR,
    MPI_LAND
  };

} // namespace MpiComm_impl

//! Construct a MpiComm for a given MPI communicator
MpiComm::MpiComm(MPI_Comm_t mpi_comm_id)
  : mpi_comm_id(mpi_comm_id)
{
  ::MPI_Comm_size( mpi_comm_id, &this->mpi_comm_size );
  ::MPI_Comm_rank( mpi_comm_id, &this->mpi_comm_rank );
}

MpiComm::MpiComm()
  : MpiComm(MPI_COMM_WORLD)
{}

void MpiComm::MPI_Barrier()
{
  ::MPI_Barrier( mpi_comm_id );
}

template<typename T>
void MpiComm::MPI_Allreduce( const T* sendbuf, T* recvbuf, int count, MPI_Op_t op ) const
{
  using namespace MpiComm_impl;
  ::MPI_Allreduce( sendbuf, recvbuf, count, mpi_type<T>(), mpi_op[op], mpi_comm_id);
}

template<typename T>
void MpiComm::MPI_Scan( const T* sendbuf, T* recvbuf, int count, MPI_Op_t op ) const
{
  using namespace MpiComm_impl;
  ::MPI_Scan( sendbuf, recvbuf, count, mpi_type<T>(), mpi_op[op], mpi_comm_id);
}

template<typename T>
void MpiComm::MPI_Allgather( const T* sendbuf, T* recvbuf, int count) const
{
  using namespace MpiComm_impl;
  ::MPI_Allgather( sendbuf, count, mpi_type<T>(),
                   recvbuf, count, mpi_type<T>(), mpi_comm_id);
}

template<typename T>
void MpiComm::MPI_Allgatherv_inplace( T* sendrecvbuf, int count) const
{
  int comm_size = this->MPI_Comm_size();

  std::vector<int> counts( comm_size );
  this->MPI_Allgather( &count, counts.data(), 1 );
  std::vector<int> displs( comm_size );
  for(int i=1; i<comm_size; i++)
    displs[i] = displs[i-1] + counts[i-1];

  using namespace MpiComm_impl;
  ::MPI_Allgatherv( MPI_IN_PLACE, 0, 0,
                    sendrecvbuf, counts.data(), displs.data(),
                    mpi_type<T>(), mpi_comm_id );
}

template<typename T>
void MpiComm::MPI_Bcast( T* buffer, int count, int root ) const
{
  using namespace MpiComm_impl;
  ::MPI_Bcast( buffer, count, mpi_type<T>(), root, mpi_comm_id);
}

template<typename T>
void MpiComm::MPI_Alltoall( const T* sendbuf, int sendcount, T* recvbuf, int recvcount ) const
{
  using namespace MpiComm_impl;
  ::MPI_Alltoall( sendbuf, sendcount, mpi_type<T>(), 
                  recvbuf, recvcount, mpi_type<T>(),
                  mpi_comm_id );
}

template<typename Kokkos_View_t>
MpiComm::MPI_Request_t MpiComm::MPI_Isend( const Kokkos_View_t& view, int dest, int tag ) const
{
  using namespace MpiComm_impl;
  MPI_Datatype type = mpi_type<typename Kokkos_View_t::value_type>();
  MPI_Request_t r = MPI_REQUEST_NULL;
  ::MPI_Isend( view.data(), view.size(), type, dest, tag, mpi_comm_id, &r);
  return r;
}

template<typename Kokkos_View_t>
MpiComm::MPI_Request_t MpiComm::MPI_Irecv( const Kokkos_View_t& view, int dest, int tag ) const
{
  using namespace MpiComm_impl;
  MPI_Datatype type = mpi_type<typename Kokkos_View_t::value_type>();
  MPI_Request_t r = MPI_REQUEST_NULL;
  ::MPI_Irecv( view.data(), view.size(), type, dest, tag, mpi_comm_id, &r);
  return r;
}

inline void MpiComm::MPI_Waitall( int count, MPI_Request_t* array_of_requests ) const
{
  ::MPI_Waitall(count, array_of_requests, MPI_STATUSES_IGNORE);
}

} // namespace dyablo