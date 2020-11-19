#include "GhostCommunicator.h"
#include <cstdint>
#include "utils/mpiUtils/GlobalMpiSession.h"

namespace dyablo{
namespace muscl_block{

GhostCommunicator_kokkos::GhostCommunicator_kokkos( std::shared_ptr<AMRmesh> amr_mesh )
{
  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
  this->nbghosts_recv = amr_mesh->getNumGhosts();

  //! Get map that contains ghost to send to each rank from PABLO : rank -> [iOcts]
  const std::map<int, std::vector<uint32_t>>& ghost_map = amr_mesh->getBordersPerProc();

  //Compute send sizes
  {
    Kokkos::realloc(this->send_sizes, nb_proc);
    this->send_sizes_host = Kokkos::create_mirror_view(this->send_sizes);  
    uint32_t nbGhostSend = 0; 
    for( auto& p : ghost_map )
    {
      int rank = p.first;
      const std::vector<uint32_t>& iOcts_source = p.second;
      send_sizes_host(rank) = iOcts_source.size();
      nbGhostSend += iOcts_source.size();
    }
    // Copy number of octants to recieve to device (host + device are up to date)
    Kokkos::deep_copy(send_sizes, send_sizes_host);

    // Allocate buffer to contain iOcts to send
    Kokkos::realloc(this->send_iOcts, nbGhostSend);
  }

  //Fill list of ghost octants to send
  {
    Kokkos::View<uint32_t*>::HostMirror send_iOcts_host = Kokkos::create_mirror_view(this->send_iOcts);
    uint32_t iOct_offset = 0;
    for( auto& p : ghost_map )
    {
      const std::vector<uint32_t>& iOcts_source = p.second;
      for( size_t i=0; i<iOcts_source.size(); i++ )
      {
        send_iOcts_host(iOct_offset + i) = iOcts_source[i];
      }
      iOct_offset += iOcts_source.size();
    }
    // Move send_iOcts to device
    Kokkos::deep_copy(this->send_iOcts, send_iOcts_host);
  }

  // Fill number of octants to recieve
  {
    Kokkos::realloc(this->recv_sizes, nb_proc);
    this->recv_sizes_host = Kokkos::create_mirror_view(this->recv_sizes);
    MPI_Alltoall( send_sizes_host.data(), 1, MPI_INT, 
                  recv_sizes_host.data(), 1, MPI_INT,
                  MPI_COMM_WORLD );
    // Copy number of octants to recieve to device (host + device are up to date)
    Kokkos::deep_copy(recv_sizes, recv_sizes_host);
  }
}

void GhostCommunicator_kokkos::exchange_ghosts(DataArray_t& U, DataArray_t& Ughost) const
{ 
  // DataArray extents
  uint32_t ncells = U.extent(0);
  uint32_t nfields = U.extent(1);

  int nb_proc = hydroSimu::GlobalMpiSession::getNProc();
  
  uint32_t nbGhostsSend = this->send_iOcts.size();
  DataArray_t send_buffers("Send buffers", ncells, nfields, nbGhostsSend);  

  const Kokkos::View<uint32_t*> & send_iOcts_ref = this->send_iOcts;
  Kokkos::parallel_for( "GhostCommunicator::fill_send_buffer", nbGhostsSend*nfields*ncells,
                        KOKKOS_LAMBDA(uint32_t index)
  {
    uint32_t iGhost = index/(nfields*ncells);
    uint32_t f = (index - iGhost*(nfields*ncells) ) / ncells ;
    uint32_t c = index - iGhost*(nfields*ncells) - f*ncells  ;

    uint32_t iOct_origin = send_iOcts_ref(iGhost);
    send_buffers(c, f, iGhost) = U(c, f, iOct_origin);
  });

  Kokkos::realloc(Ughost, ncells, nfields, this->nbghosts_recv);
  //#define MPI_IS_CUDA_AWARE
  #ifdef MPI_IS_CUDA_AWARE
  using MPIBuffer_t = DataArray_t;
  Kokkos::fence();
  MPIBuffer_t& mpi_send_buffers = send_buffers;
  // Ughost in PABLO is filled in source rank order : no need for another copy
  MPIBuffer_t& mpi_recv_buffers = Ughost;
  #else
  using MPIBuffer_t = DataArray_t::HostMirror;
  MPIBuffer_t mpi_send_buffers = Kokkos::create_mirror_view(send_buffers);
  Kokkos::deep_copy(mpi_send_buffers, send_buffers);
  MPIBuffer_t mpi_recv_buffers = Kokkos::create_mirror_view(Ughost);
  #endif

  std::vector<MPI_Request> mpi_requests;
  // Post MPI_Isends
  {
    uint32_t iOct_offset = 0;
    for(int rank=0; rank<nb_proc; rank++)
    {
      uint32_t iOct_range_end = iOct_offset+send_sizes_host(rank);
      // This only works if DataArray_t is LayoutLeft
      // Otherwise, subview is not contiguous and this is necessary for MPI
      MPIBuffer_t send_buffer_rank 
        = Kokkos::subview( mpi_send_buffers, 
                          Kokkos::ALL, 
                          Kokkos::ALL, 
                          std::make_pair(iOct_offset,iOct_range_end) );

      mpi_requests.push_back(nullptr);
      MPI_Isend( send_buffer_rank.data(), send_buffer_rank.size(), MPI_DOUBLE,
                rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );

      iOct_offset += send_sizes_host(rank);
    }
  }

  // Post MPI_Irecv to store recieved ghosts directly in Ughost
  {
    uint32_t iOct_offset = 0;
    
    for(int rank=0; rank<nb_proc; rank++)
    {
      uint32_t iOct_range_end = iOct_offset+recv_sizes_host(rank);
      // This only works if DataArray_t is LayoutLeft
      // Otherwise, subview is not contiguous and this is necessary for MPI
      MPIBuffer_t recv_buffer_rank 
        = Kokkos::subview( mpi_recv_buffers, 
                          Kokkos::ALL, 
                          Kokkos::ALL, 
                          std::make_pair(iOct_offset,iOct_range_end) );

      mpi_requests.push_back(nullptr);
      MPI_Irecv( recv_buffer_rank.data(), recv_buffer_rank.size(), MPI_DOUBLE,
                rank, 0, MPI_COMM_WORLD, &mpi_requests.back() );

      iOct_offset += recv_sizes_host(rank);
    }
  }

  MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);
  #ifdef MPI_IS_CUDA_AWARE
  Kokkos::fence();
  #else
  Kokkos::deep_copy(Ughost, mpi_recv_buffers);
  #endif
}

}//namespace muscl_block
}//namespace dyablo