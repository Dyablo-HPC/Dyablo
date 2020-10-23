/*
 * \file UserDataComm.h
 *
 * \author Pierre Kestener
 *
 * MPI ghost synchronization.
 */
#ifndef MUSCL_BLOCK_USER_DATA_COMM_H_
#define MUSCL_BLOCK_USER_DATA_COMM_H_

#include "bitpit_PABLO.hpp" // for DataCommInterface
#include "shared/kokkos_shared.h" // for type DataArray

#include "shared/FieldManager.h" // for id2index_t type

namespace dyablo { namespace muscl_block {

/**
 * Main class for user data communication for MUSCL/Hancock scheme.
 *
 * \note This class will have to be slightly refactored when
 * we will start using Kokkos/Cuda backend.
 */
class UserDataComm : public bitpit::DataCommInterface<UserDataComm>
{

public:
  using DataArray_t = DataArrayBlockHost; //! Data is host block data

  //! bulk data owned by current MPI process
  DataArray_t data;

  //! ghost data, owned by a different MPI process
  DataArray_t ghostData;
  
  //! FieldMap object for mapping field variable (ID, IP, IU, IV, ...)
  //! to actual index
  id2index_t  fm;

  //! number of scalar fields per cell
  uint32_t nbFields;

  //! number of cells per octant
  uint32_t nbCellsPerOct;

  //! number of bytes per cell to exchange
  size_t fixedSize() const;

  //! don't used, since fixed size is returning non-zero
  size_t size(const uint32_t iOct) const;

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
  UserDataComm(DataArray_t data_, DataArray_t ghostData_, id2index_t fm_);

  /**
   * Destructor.
   */
  ~UserDataComm();
  
}; // class UserDataComm

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_USER_DATA_COMM_H_
