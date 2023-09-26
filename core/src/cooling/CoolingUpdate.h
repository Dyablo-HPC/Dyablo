#pragma once

#include "cooling/CoolingUpdate_base.h"

namespace dyablo {

class CoolingUpdate_FF;

} //namespace dyablo 


template<>
inline bool dyablo::CoolingUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::CoolingUpdate_FF);

  return true;
}

