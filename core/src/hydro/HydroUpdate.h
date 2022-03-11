#pragma once

#include "HydroUpdate_base.h"

namespace dyablo {


class HydroUpdate_legacy;
class HydroUpdate_generic;
class HydroUpdate_muscl_oneneighbor;
class HydroUpdate_euler;
class HydroUpdate_RK2;

} //namespace dyablo 


template<>
inline bool dyablo::HydroUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::HydroUpdate_legacy);
  DECLARE_REGISTERED(dyablo::HydroUpdate_generic);
  DECLARE_REGISTERED(dyablo::HydroUpdate_muscl_oneneighbor);
  DECLARE_REGISTERED(dyablo::HydroUpdate_euler);
  DECLARE_REGISTERED(dyablo::HydroUpdate_RK2);

  return true;
}
