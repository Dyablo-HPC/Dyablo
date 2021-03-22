#include "GhostCommunicator.h"

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
  using DataArray_t = DataArrayBlockHost; //! Data is host block data

  //! bulk data owned by current MPI process
  DataArray_t data;

  //! ghost data, owned by a different MPI process
  DataArray_t ghostData;
  

  //! number of scalar fields per cell
  uint32_t nbFields;

  //! number of cells per octant
  uint32_t nbCellsPerOct;

  //! number of bytes per cell to exchange
  size_t fixedSize() const
  {
    return 0; 
  }

  //! don't used, since fixed size is returning non-zero
  size_t size(const uint32_t iOct) const
  {
    BITPIT_UNUSED(iOct);
    return sizeof(real_t)*nbCellsPerOct*nbFields;
  }

  /**
   * read data to communicate to neighbor MPI processes.
   */
  template<class Buffer>
  void gather(Buffer & buff, const uint32_t iOct) {

    for (uint32_t ivar=0; ivar<nbFields; ++ivar) {
      for (uint32_t index=0; index<nbCellsPerOct; ++index)
        buff << data(index,ivar,iOct);
    }

  } // gather

  /**
   * Fill ghosts with data received from neighbor MPI processes.
   */
  template<class Buffer>
  void scatter(Buffer & buff, const uint32_t iOct) {

    for (uint32_t ivar=0; ivar<nbFields; ++ivar) {
      for (uint32_t index=0; index<nbCellsPerOct; ++index)
        buff >> ghostData(index,ivar,iOct);
    }

  } // scatter

  /**
   * Constructor.
   */
  UserDataComm(DataArray_t data_, DataArray_t ghostData_) :
    data(data_),
    ghostData(ghostData_),
    nbFields(data_.extent(1)),
    nbCellsPerOct(data_.extent(0))
  {
  }; // UserDataComm::UserDataComm

  /**
   * Destructor.
   */
  ~UserDataComm(){};
  
}; // class UserDataComm

} //namespace

void GhostCommunicator_pablo::exchange_ghosts(DataArray_t& U, DataArray_t& Ughost) const
{
    uint32_t nghosts = amr_mesh->getNumGhosts();
    Kokkos::realloc(Ughost, U.extent(0), U.extent(1), nghosts);

    // Copy Data to host for MPI communication 
    DataArray_t::HostMirror U_host = Kokkos::create_mirror_view(U);
    DataArray_t::HostMirror Ughost_host = Kokkos::create_mirror_view(Ughost);
    Kokkos::deep_copy(U_host, U);

    UserDataComm data_comm(U_host, Ughost_host);
    amr_mesh->communicate(data_comm);

    // Copy back ghosts to Device
    Kokkos::deep_copy(Ughost, Ughost_host);
}

}//namespace muscl_block
}//namespace dyablo