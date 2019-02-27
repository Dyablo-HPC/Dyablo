
/*
 * UserDataComm.h for muscl scheme.
 *
 */
#ifndef MUSCL_USER_DATA_COMM_H_
#define MUSCL_USER_DATA_COMM_H_

#include "bitpit_PABLO.hpp" // for DataCommInterface
#include "shared/kokkos_shared.h" // for type DataArray

namespace euler_pablo { namespace muscl {


/**
 * Main class for user data communication.
 */
class UserDataComm : public bitpit::DataCommInterface<UserDataComm>
{

public:
  
  DataArray data;
  DataArray ghostData;

  uint32_t nbVars;
  
  size_t fixedSize() const;
  size_t size(const uint32_t e) const;

  /**
   * read data to communicate to neighbor MPI processes.
   */
  template<class Buffer>
  void gather(Buffer & buff, const uint32_t e) {

    for (uint32_t ivar=0; ivar<nbVars; ++ivar) {
      buff << data(e,ivar);
    }

  } // gather

  /**
   * Fill ghosts with data received from neighbor MPI processes.
   */
  template<class Buffer>
  void scatter(Buffer & buff, const uint32_t e) {

    for (uint32_t ivar=0; ivar<nbVars; ++ivar) {
      buff >> ghostData(e,ivar);
    }

  } // scatter

  /**
   * Constructor.
   */
  UserDataComm(DataArray data_, DataArray ghostData_);

  /**
   * Destructor.
   */
  ~UserDataComm();
  
}; // class UserDataComm

} // namespace muscl

} // namespace euler_pablo

#endif // MUSCL_USER_DATA_COMM_H_
