#pragma once

#include "IOManager_base.h"

namespace dyablo {
namespace muscl_block {

class IOManager_hdf5;

} //namespace dyablo 
} //namespace muscl_block

template<>
inline bool dyablo::muscl_block::IOManagerFactory::init()
{
  #ifdef DYABLO_USE_HDF5
  DECLARE_REGISTERED(dyablo::muscl_block::IOManager_hdf5);
  #endif
  return true;
}