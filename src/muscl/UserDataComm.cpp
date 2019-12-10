/*
 * \file UserDataComm.cpp
 *
 * \author Pierre Kestener
 */

#include "muscl/UserDataComm.h"

namespace dyablo { namespace muscl {

// ==================================================================
// ==================================================================
UserDataComm::UserDataComm(DataArray data_, DataArray ghostData_, id2index_t fm_) :
  data(data_),
  ghostData(ghostData_),
  fm(fm_),
  nbVars(data_.extent(1))
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

} // namespace dyablo
