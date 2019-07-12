/*
 * \file UserDataComm.h
 *
 * \author Pierre Kestener
 *
 */
#ifndef MUSCL_USER_DATA_COMM_H_
#define MUSCL_USER_DATA_COMM_H_

#include "bitpit_PABLO.hpp" // for DataCommInterface
#include "shared/kokkos_shared.h" // for type DataArray

#include "shared/FieldManager.h" // for id2index_t type

namespace dyablo { namespace muscl {

/**
 * Main class for user data communication for MUSCL/hancock scheme.
 *
 * \note This class will have to be slightly refactored when
 * we will start using Kokkos/Cuda backend.
 */
class UserDataComm : public bitpit::DataCommInterface<UserDataComm>
{

public:

  //! bulk data owned by current MPI process
  DataArray data;

  //! ghost data, owned by a different MPI process
  DataArray ghostData;
  
  //! FieldMap object for mapping field variable (ID, IP, IU, IV, ...)
  //! to actual index
  id2index_t  fm;

  //! number of scalar variables per cell
  uint32_t nbVars;

  //! number of bytes per cell to exchange
  size_t fixedSize() const;

  //! don't used, since fixed size is returning non-zero
  size_t size(const uint32_t e) const;

  /**
   * read data to communicate to neighbor MPI processes.
   */
  template<class Buffer>
  void gather(Buffer & buff, const uint32_t e) {

    for (uint32_t ivar=0; ivar<nbVars; ++ivar) {
      buff << data(e,fm[ivar]);
    }

  } // gather

  /**
   * Fill ghosts with data received from neighbor MPI processes.
   */
  template<class Buffer>
  void scatter(Buffer & buff, const uint32_t e) {

    for (uint32_t ivar=0; ivar<nbVars; ++ivar) {
      buff >> ghostData(e,fm[ivar]);
    }

  } // scatter

  /**
   * Constructor.
   */
  UserDataComm(DataArray data_, DataArray ghostData_, id2index_t fm_);

  /**
   * Destructor.
   */
  ~UserDataComm();
  
}; // class UserDataComm

} // namespace muscl

} // namespace dyablo

#endif // MUSCL_USER_DATA_COMM_H_
