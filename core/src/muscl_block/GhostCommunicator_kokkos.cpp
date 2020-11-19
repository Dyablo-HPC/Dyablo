#include "GhostCommunicator.h"
#include <cstdint>
#include "utils/mpiUtils/GlobalMpiSession.h"

namespace dyablo{
namespace muscl_block{

GhostCommunicator_kokkos::GhostCommunicator_kokkos( std::shared_ptr<AMRmesh> amr_mesh )
{
  constexpr int GHOST_COMM_TAG = 951;

  this->nbghosts = amr_mesh->getNumGhosts();
  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();  

  std::vector<int> send_sizes(nb_proc);
  {
    //! Map that contains ghost to send to each rank : rank -> [iOcts]
    std::map<int, std::vector<uint32_t>> ghost_map = amr_mesh->getBordersPerProc();
    this->ioct_send_matrix = std::vector<std::vector<uint32_t>>(nb_proc);
    for( auto& p : ghost_map )
    {
      int rank = p.first;
      std::vector<uint32_t>& iOcts_source = p.second;
      ioct_send_matrix[rank] = iOcts_source;
      send_sizes[rank] = iOcts_source.size();
    }
  }
  this->recv_sizes = std::vector<int> (nb_proc);
  {
    MPI_Alltoall( send_sizes.data(), 1, MPI_INT, 
                  recv_sizes.data(), 1, MPI_INT,
                  MPI_COMM_WORLD );
  }

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
    Kokkos::realloc(recv_buffers[rank], ncells, nfields, recv_sizes[rank]);
    mpi_requests.push_back(nullptr);
    MPI_Irecv( recv_buffers[rank].data(), recv_buffers[rank].size(), MPI_DOUBLE,
               rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );
  }

  MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);

  uint32_t iOct_offset = 0;
  for(int rank=0; rank<nb_proc; rank++)
  {
    for( int i=0; i<recv_sizes[rank]; i++  )
    {
      uint32_t iOct_ghost = iOct_offset + i;
      for (uint32_t f=0; f<nfields; ++f) {
        for (uint32_t c=0; c<ncells; ++c)
          Ughost_host(c, f, iOct_ghost) = recv_buffers[rank](c, f, i);
      }      
    }
    iOct_offset+=recv_sizes[rank];
  }

  Kokkos::deep_copy(Ughost, Ughost_host);
}

}//namespace muscl_block
}//namespace dyablo