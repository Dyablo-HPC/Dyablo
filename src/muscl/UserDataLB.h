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

namespace dyablo { namespace muscl {

/**
 * Main class for user data load balancing.
 */
class UserDataLB : public bitpit::DataLBInterface<UserDataLB>
{

public:
  
  // pass by copy (Kokkos::View), watchout data and ghostdata
  // wii surely be reassigned, so calling code must be aware of
  // that.
  DataArray data;
  DataArray ghostdata;

  uint32_t nbVars;
  
  size_t fixedSize() const;
  size_t size(const uint32_t e) const;
  void move(const uint32_t from, const uint32_t to);

  template<class Buffer>
  void gather(Buffer & buff, const uint32_t e) {
    for (uint32_t ivar=0; ivar<nbVars; ++ivar)
      buff << data(e,ivar);
  };

  template<class Buffer>
  void scatter(Buffer & buff, const uint32_t e) {
    for (uint32_t ivar=0; ivar<nbVars; ++ivar)
      buff >> data(e,ivar);
  };

  void assign(uint32_t stride, uint32_t length);
  void resize(uint32_t newSize);
  void resizeGhost(uint32_t newSize);
  void shrink();

  UserDataLB(DataArray data_, DataArray ghostdata_);
  ~UserDataLB();
  
}; // class UserDataLb

} // namespace muscl

} // namespace dyablo

#endif // MUSCL_USER_DATA_LB_H_
