#include "GhostCommunicator.h"
#include <cstdint>
#include "utils/mpi/GlobalMpiSession.h"

namespace dyablo{

void GhostCommunicator_kokkos::private_init_domains(const Kokkos::View< int* > domains)
{
  int mpi_size = mpi_comm.MPI_Comm_size();

  this->send_sizes = Kokkos::View<uint32_t*>( "GhostCommunicator_kokkos::send_sizes", mpi_size );
  this->recv_sizes = Kokkos::View<uint32_t*>( "GhostCommunicator_kokkos::recv_sizes", mpi_size );
  this->send_sizes_host = Kokkos::create_mirror_view( send_sizes );
  this->recv_sizes_host = Kokkos::create_mirror_view( recv_sizes );
  this->send_iOcts = Kokkos::View<uint32_t*>( "GhostCommunicator_kokkos::send_iOcts", domains.size() );

  // Local refs to avoid dereferencing this in kernels
  Kokkos::View<uint32_t*>& send_sizes = this->send_sizes; 
  Kokkos::View<uint32_t*>& send_iOcts = this->send_iOcts; 

  // Compute send_sizes from domain list 
  Kokkos::parallel_for( "GhostCommunicator_kokkos::construct_from_domain::count_send_sizes", 
    domains.size(),
    KOKKOS_LAMBDA( int i )
  {
    Kokkos::atomic_increment( &send_sizes( domains(i) ) );
  });
  Kokkos::deep_copy( send_sizes_host, send_sizes );
  
  // Fill send_iOcts
  {
    // Compute offset for the beginning of each domain
    Kokkos::View<uint32_t*> rank_offset( "rank_offset", mpi_size );
    Kokkos::parallel_scan( "GhostCommunicator_kokkos::construct_from_domain::rank_offset", 
      mpi_size,
      KOKKOS_LAMBDA( int rank, int& offset, bool final )
    {
      if(final) rank_offset(rank) = offset;
      offset += send_sizes(rank);
    });

    Kokkos::parallel_for( "GhostCommunicator_kokkos::construct_from_domain::compute_send_iOcts",
      domains.size(),
      KOKKOS_LAMBDA( int i )
    {
      int domain = domains(i);
      uint32_t iPack = Kokkos::atomic_fetch_inc( &rank_offset( domain ) );
      send_iOcts(iPack) = i;
    });
    // TODO : maybe sort to avoid random read order
  }

  // Exchange sizes to send/recieve  
  mpi_comm.MPI_Alltoall( send_sizes_host.data(), 1, recv_sizes_host.data(), 1 );  
  this->nbghosts_recv = 0;
  for( int r=0; r<mpi_size; r++ )
    this->nbghosts_recv += recv_sizes_host(r);
  Kokkos::deep_copy( recv_sizes, recv_sizes_host );
}

/**
 * Initialize GhostCommunicator_kokkos with the list of octants to send as ghosts for each rank
 * @param send_sizes nomber of octants to send to rank i
 * @param send_iOcts octants to send to rank 0, then to rank 1, etc (of size sum(send_sizes))
 **/
void GhostCommunicator_kokkos::private_init_map( const Kokkos::View< uint32_t* > send_sizes, const Kokkos::View< uint32_t* > send_iOcts )
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

GhostCommunicator_kokkos::GhostCommunicator_kokkos( const std::map<int, std::vector<uint32_t>>& ghost_map, const MpiComm& mpi_comm )
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

uint32_t GhostCommunicator_kokkos::getNumGhosts() const
{
  return this->nbghosts_recv;
}


}//namespace dyablo
