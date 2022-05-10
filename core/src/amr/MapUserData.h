#pragma once

#include "MapUserData_base.h"

namespace dyablo {


class MapUserData_legacy;

} //namespace dyablo 


template<>
inline bool dyablo::MapUserDataFactory::init()
{
  DECLARE_REGISTERED(dyablo::MapUserData_legacy);

  return true;
}