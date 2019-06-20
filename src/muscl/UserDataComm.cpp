/*
 * \file UserDataComm.cpp
 *
 * \author Pierre Kestener
 */

#include "muscl/UserDataComm.h"

namespace euler_pablo { namespace muscl {

// ==================================================================
// ==================================================================
UserDataComm::UserDataComm(DataArray data_, DataArray ghostData_, id2index_t fm_) :
  data(data_),
  ghostData(ghostData_),
  fm(fm_),
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
  
  return sizeof(real_t)*nbVars;
  
}; // UserDataComm::fixedSize

// ==================================================================
// ==================================================================
size_t UserDataComm::size(const uint32_t e) const
{
  
  BITPIT_UNUSED(e);
  return 0;
 
}; // UserDataComm::size

} // namespace muscl

} // namespace euler_pablo
