/*
 * \file UserDataLB.cpp
 *
 * \author Pierre Kestener
 *
 */

#include "muscl/UserDataLB.h"

#include "bitpit_common.hpp"

namespace dyablo { namespace muscl {

// ==================================================================
// ==================================================================
UserDataLB::UserDataLB(DataArray data_, DataArray ghostdata_) :
  data(data_),
  ghostdata(ghostdata_),
  nbVars(data_.dimension(1))
{
};

// ==================================================================
// ==================================================================
UserDataLB::~UserDataLB()
{
};

// ==================================================================
// ==================================================================
size_t UserDataLB::fixedSize() const
{
  return 0;
};

// ==================================================================
// ==================================================================
size_t UserDataLB::size(const uint32_t e) const
{
  BITPIT_UNUSED(e);
  return sizeof(double)*nbVars;
};

// ==================================================================
// ==================================================================
void UserDataLB::move(const uint32_t from, const uint32_t to)
{
  for (uint32_t ivar=0; ivar<nbVars; ++ivar)
    data(to,ivar) = data(from,ivar);
};

// ==================================================================
// ==================================================================
void UserDataLB::assign(uint32_t stride, uint32_t length)
{
  
  DataArray dataCopy("dataLBcopy");
  Kokkos::resize(dataCopy,length,nbVars);
  
  Kokkos::parallel_for(length, KOKKOS_LAMBDA(size_t &i) {
      for (uint32_t ivar=0; ivar<nbVars; ++ivar)
	dataCopy(i,ivar) = data(i+stride,ivar);
    });
  
  data = dataCopy;
};

// ==================================================================
// ==================================================================
void UserDataLB::resize(uint32_t newSize)
{
  Kokkos::resize(data,newSize,nbVars);
};

// ==================================================================
// ==================================================================
void UserDataLB::resizeGhost(uint32_t newSize)
{
  Kokkos::resize(ghostdata,newSize,nbVars);
};

// ==================================================================
// ==================================================================
void UserDataLB::shrink() {
  // TODO ?
};

} // namespace muscl

} // namespace dyablo
