#include "ViewCommunicator.h"
#include <cstdint>
#include "utils/mpi/GlobalMpiSession.h"

namespace dyablo{

void ViewCommunicator::private_init_domains(const Kokkos::View< int* > domains)
{
  int mpi_size = mpi_comm.MPI_Comm_size();

  this->send_sizes = Kokkos::View<uint32_t*>( "ViewCommunicator::send_sizes", mpi_size );
  this->recv_sizes = Kokkos::View<uint32_t*>( "ViewCommunicator::recv_sizes", mpi_size );
  this->send_sizes_host = Kokkos::create_mirror_view( send_sizes );
  this->recv_sizes_host = Kokkos::create_mirror_view( recv_sizes );
  this->send_iOcts = Kokkos::View<uint32_t*>( "ViewCommunicator::send_iOcts", domains.size() );

  // Local refs to avoid dereferencing this in kernels
  Kokkos::View<uint32_t*>& send_sizes = this->send_sizes; 
  Kokkos::View<uint32_t*>& send_iOcts = this->send_iOcts; 

  // Compute send_sizes from domain list 
  Kokkos::parallel_for( "ViewCommunicator::construct_from_domain::count_send_sizes", 
    domains.size(),
    KOKKOS_LAMBDA( int i )
  {
    Kokkos::atomic_increment( &send_sizes( domains(i) ) );
  });
  Kokkos::deep_copy( send_sizes_host, send_sizes );
  
  // Fill send_iOcts
  {
    uint32_t offset = 0;
    for(int rank=0; rank<mpi_size; rank++)
    {
      if( send_sizes_host(rank) > 0 )
      {
        Kokkos::parallel_scan( "ViewCommunicator::construct_from_domain::compute_send_iOcts",
          domains.size(),
          KOKKOS_LAMBDA( uint32_t iOct, uint32_t& i_write, bool final )
        {
          if( domains(iOct) == rank )
          {
            if( final )
              send_iOcts( offset + i_write ) = iOct;
            i_write++;
          }
        });
        offset += send_sizes_host(rank);
      }
    }
  }

  // Exchange sizes to send/recieve  
  mpi_comm.MPI_Alltoall( send_sizes_host.data(), 1, recv_sizes_host.data(), 1 );  
  this->nbghosts_recv = 0;
  for( int r=0; r<mpi_size; r++ )
    this->nbghosts_recv += recv_sizes_host(r);
  Kokkos::deep_copy( recv_sizes, recv_sizes_host );
}

/// See ViewCommunicator's related constructor
void ViewCommunicator::private_init_map( const Kokkos::View< uint32_t* > send_sizes, const Kokkos::View< uint32_t* > send_iOcts )
{
  int nb_proc = mpi_comm.MPI_Comm_size();

  this->send_sizes = send_sizes;
  this->send_iOcts = send_iOcts;
  this->send_sizes_host = Kokkos::create_mirror_view( send_sizes );
  Kokkos::deep_copy( this->send_sizes_host, send_sizes );

  // Fill number of octants to recieve
  {
    Kokkos::realloc(this->recv_sizes, nb_proc);
    this->recv_sizes_host = Kokkos::create_mirror_view(this->recv_sizes);
    mpi_comm.MPI_Alltoall(  send_sizes_host.data(), 1, 
                            recv_sizes_host.data(), 1);
    // Copy number of octants to recieve to device (host + device are up to date)
    Kokkos::deep_copy(recv_sizes, recv_sizes_host);
    this->nbghosts_recv = 0;
    for(int i=0; i<nb_proc; i++)
      this->nbghosts_recv += recv_sizes_host(i);
  }

}

ViewCommunicator::ViewCommunicator( const std::map<int, std::vector<uint32_t>>& ghost_map, const MpiComm& mpi_comm )
  : mpi_comm(mpi_comm)
{
  int nb_proc = mpi_comm.MPI_Comm_size();

  // Allocate send_sizes
  Kokkos::View< uint32_t* > send_sizes("send_sizes", nb_proc);
  auto send_sizes_host = Kokkos::create_mirror_view(send_sizes);  
  
  //Compute send sizes (on Host)
  uint32_t nbGhostSend = 0; 
  for(int rank=0; rank<nb_proc; rank++)
  {
    auto it = ghost_map.find(rank);
    if(it==ghost_map.end())
    {
      send_sizes_host(rank) = 0;
    }
    else
    {
      const std::vector<uint32_t>& iOcts_source = it->second;
      send_sizes_host(rank) = iOcts_source.size();
      nbGhostSend += iOcts_source.size();
    }
  }
  // Synchronize send sizes to device
  Kokkos::deep_copy(send_sizes, send_sizes_host);

  // Allocate send_iOcts (on Device)
  Kokkos::View< uint32_t* > send_iOcts("send_iOcts", nbGhostSend);
  uint32_t offset = 0;
  for(int rank=0; rank<nb_proc; rank++)
  {
    uint32_t send_size_rank = send_sizes_host(rank);
    if( send_size_rank > 0 )
    {
      // Create an unmanaged view from ghost_map[rank]
      const std::vector<uint32_t>& iOcts_rank_vector = ghost_map.at(rank);
      using UnmanagedHostView = Kokkos::View<const uint32_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
      UnmanagedHostView iOcts_rank_host( iOcts_rank_vector.data(), iOcts_rank_vector.size() );
      
      // Create a subview of send_iOcts for iOcts to send to `rank`
      auto iOcts_rank_device = Kokkos::subview( send_iOcts, std::pair( offset, offset+send_size_rank ) );
      
      // Copy octants from ghost_map[rank] to send_iOcts
      Kokkos::deep_copy( iOcts_rank_device, iOcts_rank_host );
    }
    offset += send_size_rank;
  }

  private_init_map(send_sizes, send_iOcts);
}

uint32_t ViewCommunicator::getNumGhosts() const
{
  return this->nbghosts_recv;
}


}//namespace dyablo
