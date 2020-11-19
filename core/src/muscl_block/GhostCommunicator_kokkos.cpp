#include "GhostCommunicator.h"
#include <cstdint>
#include "utils/mpiUtils/GlobalMpiSession.h"

namespace dyablo{
namespace muscl_block{

namespace {

/**
 * Callback class to serialize/deserialize user data for PABLO MPI communication
 * (block based version)
 */
class UserDataComm : public bitpit::DataCommInterface<UserDataComm>
{

public:
  using Ghost_id_view_t = Kokkos::View<uint32_t*>;

  UserDataComm(std::vector<std::vector<uint32_t>>& ioct_recv_matrix, std::vector<std::vector<uint32_t>>& ioct_send_matrix)
  : ioct_recv_matrix(ioct_recv_matrix),
    ioct_send_matrix(ioct_send_matrix)
  {
  }

  /**
   * read data to communicate to neighbor MPI processes.
   */
  template<class Buffer>
  void gather(Buffer & buff, int rank, const uint32_t iOct_origin) {
    ioct_send_matrix[rank].push_back(iOct_origin);
    buff << iOct_origin;
  } // gather

  /**
   * Fill ghosts with data received from neighbor MPI processes.
   */
  template<class Buffer>
  void scatter(Buffer & buff, int rank, const uint32_t iOct_dest) {
    uint32_t iOct_origin;
    buff >> iOct_origin;
    ioct_recv_matrix[rank].push_back(iOct_dest);
  } // scatter

  std::vector<std::vector<uint32_t>>& ioct_recv_matrix; //! Octants recieved from each rank
  std::vector<std::vector<uint32_t>>& ioct_send_matrix; //! Octants send to each rank
}; // class UserDataComm

} //namespace

GhostCommunicator_kokkos::GhostCommunicator_kokkos( std::shared_ptr<AMRmesh> amr_mesh )
{
  using View_t = UserDataComm::Ghost_id_view_t; 
  
  this->nbghosts = amr_mesh->getNumGhosts();

  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
  ioct_recv_matrix = std::vector<std::vector<uint32_t>>(nb_proc);
  ioct_send_matrix = std::vector<std::vector<uint32_t>>(nb_proc);

  UserDataComm comm(ioct_recv_matrix, ioct_send_matrix);
  amr_mesh->com_metadata(comm);
}

void GhostCommunicator_kokkos::exchange_ghosts(DataArray_t& U, DataArray_t& Ughost) const
{ 
  
  uint32_t ncells = U.extent(0);
  uint32_t nfields = U.extent(1);
  
  auto U_host = Kokkos::create_mirror_view(U);

  Kokkos::realloc(Ughost, ncells, nfields, this->nbghosts);
  auto Ughost_host = Kokkos::create_mirror_view(Ughost);

  Kokkos::deep_copy(U_host, U);

  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
  std::vector<DataArray_t::HostMirror> send_buffers(nb_proc);
  std::vector<MPI_Request> mpi_requests;
  for(int rank=0; rank<nb_proc; rank++)
  {
    Kokkos::realloc(send_buffers[rank], ncells, nfields, ioct_send_matrix[rank].size());

    for( size_t i=0; i<ioct_send_matrix[rank].size(); i++  )
    {
      uint32_t iOct_origin = ioct_send_matrix[rank][i];
      for (uint32_t f=0; f<nfields; ++f) {
        for (uint32_t c=0; c<ncells; ++c)
          send_buffers[rank](c, f, i) = U_host(c, f, iOct_origin);
      }      
    }
    mpi_requests.push_back(nullptr);
    MPI_Isend( send_buffers[rank].data(), send_buffers[rank].size(), MPI_DOUBLE,
               rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );
  }

  std::vector<DataArray_t::HostMirror> recv_buffers(nb_proc);
  for(int rank=0; rank<nb_proc; rank++)
  {
    Kokkos::realloc(recv_buffers[rank], ncells, nfields, ioct_recv_matrix[rank].size());
    mpi_requests.push_back(nullptr);
    MPI_Irecv( recv_buffers[rank].data(), recv_buffers[rank].size(), MPI_DOUBLE,
               rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );
  }

  MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);

  for(int rank=0; rank<nb_proc; rank++)
  {
    for( size_t i=0; i<ioct_recv_matrix[rank].size(); i++  )
    {
      uint32_t iOct_ghost = ioct_recv_matrix[rank][i];
      for (uint32_t f=0; f<nfields; ++f) {
        for (uint32_t c=0; c<ncells; ++c)
          Ughost_host(c, f, iOct_ghost) = recv_buffers[rank](c, f, i);
      }      
    }
  }

  Kokkos::deep_copy(Ughost, Ughost_host);
}

}//namespace muscl_block
}//namespace dyablo