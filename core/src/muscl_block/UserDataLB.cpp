/*
 * \file UserDataLB.cpp
 *
 * \author Pierre Kestener
 *
 */

#include "muscl_block/UserDataLB.h"

#include "bitpit_common.hpp"

namespace dyablo { namespace muscl_block {

// ==================================================================
// ==================================================================
UserDataLB::UserDataLB(DataArrayBlock& data_, 
                       DataArrayBlock& ghostdata_, 
                       id2index_t fm_) :
  data(data_),
  ghostdata(ghostdata_),
  fm(fm_),
  nbFields(data_.extent(1)),
  nbCellsPerOct(data_.extent(0))
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
size_t UserDataLB::size(const uint32_t iOct) const
{

  BITPIT_UNUSED(iOct);

  // return size per octant
  return sizeof(real_t)*nbCellsPerOct*nbFields;

}; // UserDataLB::size

// ==================================================================
// ==================================================================
void UserDataLB::move(const uint32_t from, const uint32_t to)
{

  for (uint32_t ivar=0; ivar<nbFields; ++ivar)
    for (uint32_t index=0; index<nbCellsPerOct; ++index)
      data(index,fm[ivar],to) = data(index,fm[ivar],from);

}; // UserDataLB::move

// ==================================================================
// ==================================================================
void UserDataLB::assign(uint32_t stride, uint32_t length)
{
  
  DataArrayBlock dataCopy("dataLBcopy");
  Kokkos::resize(dataCopy, nbCellsPerOct, length, nbFields);
  
  Kokkos::parallel_for("dyablo::muscl_block::UserDataLB::assign copy data to dataCopy",length, KOKKOS_LAMBDA(size_t &iOct) {
      for (uint32_t ivar=0; ivar<nbFields; ++ivar)
        for (uint32_t index=0; index<nbCellsPerOct; ++index)
          dataCopy(index, fm[ivar], iOct) = data(index, fm[ivar], iOct+stride);
    });
  
  //data = dataCopy;
  //Kokkos::resize(data,nbCellsPerOct,length,nbVars);
  Kokkos::parallel_for("dyablo::muscl_block::UserDataLB::assign copy dataCopy to data",length, KOKKOS_LAMBDA(size_t &iOct) {
      for (uint32_t ivar=0; ivar<nbFields; ++ivar)
        for (uint32_t index=0; index<nbCellsPerOct; ++index)
          data(index,fm[ivar],iOct) = dataCopy(index,fm[ivar],iOct);
    });

}; // UserDataLB::assign

// ==================================================================
// ==================================================================
void UserDataLB::resize(uint32_t newSize)
{

  Kokkos::resize(data, nbCellsPerOct, nbFields, newSize);

}; // UserDataLB::resize

// ==================================================================
// ==================================================================
void UserDataLB::resizeGhost(uint32_t newSize)
{

  Kokkos::resize(ghostdata, nbCellsPerOct, nbFields, newSize);

}; // UserDataLB::resizeGhost

// ==================================================================
// ==================================================================
void UserDataLB::shrink() {
  
  // TODO ?

}; // UserDataLB::shrink

} // namespace muscl_block

} // namespace dyablo
