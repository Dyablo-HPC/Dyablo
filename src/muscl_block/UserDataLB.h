/*
 * \file UserDataLB.h for muscl scheme.
 *
 * \author Pierre Kestener
 *
 * MPI load balancing user data.
 */
#ifndef MUSCL_USER_DATA_LB_H_
#define MUSCL_USER_DATA_LB_H_

#include <cstdlib>
#include <cstdint>

#include "DataLBInterface.hpp" // for DataCommInterface
#include "shared/kokkos_shared.h"

#include "shared/FieldManager.h" // for id2index_t type

namespace dyablo { namespace muscl_block {

/**
 * Main class for user data load balancing for MUSCL/Hancock scheme.
 *
 * \note This class will have to be slightly refactored when
 * we will start using Kokkos/Cuda backend.
 */
class UserDataLB : public bitpit::DataLBInterface<UserDataLB>
{

public:
  
  // pass by copy (Kokkos::View), watchout data and ghostdata
  // will surely be reassigned, so calling code must be aware of
  // that.
  // remember DataArrayBlock is 3d :
  // - first dim is cell id inside block
  // - second dim is variable (rho, rhov, energy, ...)
  // - third dim is octant id
  DataArrayBlock& data;
  DataArrayBlock& ghostdata;

  //! FieldMap object for mapping field variable (ID, IP, IU, IV, ...)
  //! to actual index
  id2index_t  fm;

  //! number of scalar variables per cell
  uint32_t nbVars;

  //! number of cells per octant
  uint32_t nbCellsPerOct;

  //! number of bytes per cell to exchange
  size_t fixedSize() const;
  
  //! number of bytes per cell to exchange
  size_t size(const uint32_t e) const;

  //! move user data from one cell to another
  void move(const uint32_t from, const uint32_t to);

  //! read data to be re-assigned to another MPI process
  template<class Buffer>
  void gather(Buffer & buff, const uint32_t iOct) {
    // copy block of data
    for (uint32_t index=0; index<nbCellsPerOct; ++index)
      for (uint32_t ivar=0; ivar<nbVars; ++ivar)
        buff << data(index, fm[ivar], iOct);
  };

  //! write data received from another MPI process
  template<class Buffer>
  void scatter(Buffer & buff, const uint32_t iOct) {
    // copy block of data
    for (uint32_t index=0; index<nbCellsPerOct; ++index)
      for (uint32_t ivar=0; ivar<nbVars; ++ivar)
        buff >> data(index, fm[ivar], iOct);
  };

  /**
   * extract the range of data [stride, stride+length[ and
   * re-assign DataArrayBlock data
   */
  void assign(uint32_t stride, uint32_t length);

  //! resize DataArrayBlock data
  void resize(uint32_t newSize);

  //! resize DataArrayBlock ghostdata
  void resizeGhost(uint32_t newSize);

  //! not sure we really need it
  void shrink();

  /**
   * Constructor.
   */
  UserDataLB(DataArrayBlock& data_, DataArrayBlock& ghostdata_, id2index_t fm_);
  
  /**
   * Destructor.
   */
  ~UserDataLB();
  
}; // class UserDataLB

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_USER_DATA_LB_H_
