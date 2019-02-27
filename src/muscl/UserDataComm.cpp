/*
 * UserDataComm.cpp
 *
 */

#include "muscl/UserDataComm.h"

namespace euler_pablo { namespace muscl {

// ==================================================================
// ==================================================================
UserDataComm::UserDataComm(DataArray data_, DataArray ghostData_) :
  data(data_),
  ghostData(ghostData_),
  nbVars(data_.dimension(1))
{
}; // UserDataComm::UserDataComm

// ==================================================================
// ==================================================================
UserDataComm::~UserDataComm()
{
}; // UserDataComm::~UserDataComm

// ==================================================================
// ==================================================================
size_t UserDataComm::fixedSize() const
{
  
  return 0;
  
}; // UserDataComm::fixedSize

// ==================================================================
// ==================================================================
size_t UserDataComm::size(const uint32_t e) const
{
  
  BITPIT_UNUSED(e);
  
  return sizeof(real_t)*nbVars;
  
}; // UserDataComm::size

} // namespace muscl

} // namespace euler_pablo
