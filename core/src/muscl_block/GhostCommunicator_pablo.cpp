#include "GhostCommunicator.h"

namespace dyablo{
namespace muscl_block{

namespace {

/**
 * Callback class to serialize/deserialize user data for PABLO MPI communication
 * (block based version)
 */
template<typename DataArray_t>
class UserDataComm : public bitpit::DataCommInterface<UserDataComm<DataArray_t>>
{

public:
  //! bulk data owned by current MPI process
  DataArray_t data;

  //! ghost data, owned by a different MPI process
  DataArray_t ghostData;

  size_t elts_per_oct;

  UserDataComm(DataArray_t data_, DataArray_t ghostData_) :
    data(data_),
    ghostData(ghostData_)
  {
    elts_per_oct = 1;
    for( int i=0; i<DataArray_t::rank-1; i++ )
    {
      elts_per_oct *= data.extent(i);
    }
  }; // UserDataComm::UserDataComm
  

  //! number of bytes per cell to exchange
  size_t fixedSize() const
  {
    return sizeof(typename DataArray_t::value_type)*elts_per_oct;
  }

  /// Fill MPI buffers with data to send to neighbor MPI processes.
  template<class Buffer>
  std::enable_if_t< DataArray_t::rank == 3, void> 
  gather(Buffer & buff, const uint32_t iOct) {
    for(uint32_t i1=0; i1<data.extent(1); i1++)
      for(uint32_t i0=0; i0<data.extent(0); i0++)
      {
        buff << data( i0, i1, iOct );
      }
  }

  /// Fill ghosts with data received from neighbor MPI processes.
  template<class Buffer>
  std::enable_if_t< DataArray_t::rank == 3, void> 
  scatter(Buffer & buff, const uint32_t iOct) {
    for(uint32_t i1=0; i1<data.extent(1); i1++)
      for(uint32_t i0=0; i0<data.extent(0); i0++)
      {
        buff >> data( i0, i1, iOct );
      }
  }

  ~UserDataComm(){};
  
}; // class UserDataComm

template< typename DataArray_t >
void exchange_ghosts_aux( const std::shared_ptr<AMRmesh>& amr_mesh, 
                          const DataArray_t& U, DataArray_t& Ughost)
{
  assert(U.extent( DataArray_t::rank-1 ) == amr_mesh->getNumOctants()); // Last index must be iOct

  uint32_t nghosts = amr_mesh->getNumGhosts();
  Kokkos::realloc(Ughost, U.extent(0), U.extent(1), nghosts);

  // Copy Data to host for MPI communication 
  typename DataArray_t::HostMirror U_host = Kokkos::create_mirror_view(U);
  typename DataArray_t::HostMirror Ughost_host = Kokkos::create_mirror_view(Ughost);
  Kokkos::deep_copy(U_host, U);

  UserDataComm<DataArray_t> data_comm(U_host, Ughost_host);
  amr_mesh->communicate(data_comm);

  // Copy back ghosts to Device
  Kokkos::deep_copy(Ughost, Ughost_host);
}


} //namespace

void GhostCommunicator_pablo::exchange_ghosts(const DataArrayBlock& U, DataArrayBlock& Ughost) const
{
    exchange_ghosts_aux(amr_mesh, U, Ughost);
}

}//namespace muscl_block
}//namespace dyablo