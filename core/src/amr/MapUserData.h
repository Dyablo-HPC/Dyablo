#pragma once

#include "MapUserData_base.h"

namespace dyablo {


class MapUserData_legacy;
class MapUserData_mean;

} //namespace dyablo 


template<>
inline bool dyablo::MapUserDataFactory::init()
{
  //DECLARE_REGISTERED(dyablo::MapUserData_legacy);
  DECLARE_REGISTERED(dyablo::MapUserData_mean);

  return true;
}