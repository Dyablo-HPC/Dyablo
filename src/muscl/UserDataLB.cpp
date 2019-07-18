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
UserDataLB::UserDataLB(DataArray& data_, 
                       DataArray& ghostdata_, 
                       id2index_t fm_) :
  data(data_),
  ghostdata(ghostdata_),
  fm(fm_),
  nbVars(data_.dimension(1))
{
}; // UserDataLB::UserDataLB

// ==================================================================
// ==================================================================
UserDataLB::~UserDataLB()
{
}; // ~UserDataLB::UserDataLB

// ==================================================================
// ==================================================================
size_t UserDataLB::fixedSize() const
{

  return 0;

}; // UserDataLB::fixedSize

// ==================================================================
// ==================================================================
size_t UserDataLB::size(const uint32_t e) const
{

  BITPIT_UNUSED(e);
  return sizeof(real_t)*nbVars;

}; // UserDataLB::size

// ==================================================================
// ==================================================================
void UserDataLB::move(const uint32_t from, const uint32_t to)
{

  for (uint32_t ivar=0; ivar<nbVars; ++ivar)
    data(to,fm[ivar]) = data(from,fm[ivar]);

}; // UserDataLB::move

// ==================================================================
// ==================================================================
void UserDataLB::assign(uint32_t stride, uint32_t length)
{
  
  DataArray dataCopy("dataLBcopy");
  Kokkos::resize(dataCopy,length,nbVars);
  
  Kokkos::parallel_for(length, KOKKOS_LAMBDA(size_t &i) {
      for (uint32_t ivar=0; ivar<nbVars; ++ivar)
	dataCopy(i,fm[ivar]) = data(i+stride,fm[ivar]);
    });
  
  //data = dataCopy;
  //Kokkos::resize(data,length,nbVars);
  Kokkos::parallel_for(length, KOKKOS_LAMBDA(size_t &i) {
      for (uint32_t ivar=0; ivar<nbVars; ++ivar)
        data(i,fm[ivar]) = dataCopy(i,fm[ivar]);
    });

}; // UserDataLB::assign

// ==================================================================
// ==================================================================
void UserDataLB::resize(uint32_t newSize)
{

  Kokkos::resize(data,newSize,nbVars);

}; // UserDataLB::resize

// ==================================================================
// ==================================================================
void UserDataLB::resizeGhost(uint32_t newSize)
{

  Kokkos::resize(ghostdata,newSize,nbVars);

}; // UserDataLB::resizeGhost

// ==================================================================
// ==================================================================
void UserDataLB::shrink() {
  
  // TODO ?

}; // UserDataLB::shrink

} // namespace muscl

} // namespace dyablo
