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
UserDataLB::UserDataLB(DataArrayHost& data_, 
                       DataArrayHost& ghostdata_ )
  :
  data(data_),
  ghostdata(ghostdata_),
  nbVars(data_.extent(1))
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
    data(to,ivar) = data(from,ivar);

}; // UserDataLB::move

// ==================================================================
// ==================================================================
void UserDataLB::assign(uint32_t stride, uint32_t length)
{
  
  DataArrayHost dataCopy("dataLBcopy");
  Kokkos::resize(dataCopy,length,nbVars);
  
  Kokkos::parallel_for("dyablo::muscl::UserDataLB::assign copy data to dataCopy",
    Kokkos::RangePolicy<Kokkos::OpenMP>( 0, length ), 
    [=](size_t &i) {
      for (uint32_t ivar=0; ivar<nbVars; ++ivar)
	      dataCopy(i,ivar) = data(i+stride,ivar);
    });
  
  //data = dataCopy;
  //Kokkos::resize(data,length,nbVars);
  Kokkos::parallel_for("dyablo::muscl::UserDataLB::assign copy dataCopy to data",
      Kokkos::RangePolicy<Kokkos::OpenMP>( 0, length ), 
      [=](size_t &i) {
        for (uint32_t ivar=0; ivar<nbVars; ++ivar)
          data(i,ivar) = dataCopy(i,ivar);
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
