#pragma once

#include "IOManager_base.h"

namespace dyablo {


class IOManager_hdf5;

} //namespace dyablo 


template<>
inline bool dyablo::IOManagerFactory::init()
{
  #ifdef DYABLO_USE_HDF5
  DECLARE_REGISTERED(dyablo::IOManager_hdf5);
  #endif
  return true;
}