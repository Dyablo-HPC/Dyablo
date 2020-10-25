/*
 * \file UserDataComm.cpp
 *
 * \author Pierre Kestener
 */

#include "muscl_block/UserDataComm.h"

namespace dyablo { namespace muscl_block {

// ==================================================================
// ==================================================================
UserDataComm::UserDataComm(DataArray_t data_, DataArray_t ghostData_, id2index_t fm_) :
  data(data_),
  ghostData(ghostData_),
  fm(fm_),
  nbFields(data_.extent(1)),
  nbCellsPerOct(data_.extent(0))
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
size_t UserDataComm::size(const uint32_t iOct) const
{
  
  BITPIT_UNUSED(iOct);
  return sizeof(real_t)*nbCellsPerOct*nbFields;
 
}; // UserDataComm::size

} // namespace muscl

} // namespace dyablo
