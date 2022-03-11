#pragma once

#include "HydroUpdate_base.h"

namespace dyablo {


class HydroUpdate_legacy;
class HydroUpdate_generic;
class HydroUpdate_muscl_oneneighbor;
class HydroUpdate_euler;

} //namespace dyablo 


template<>
inline bool dyablo::HydroUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::HydroUpdate_legacy);
  DECLARE_REGISTERED(dyablo::HydroUpdate_generic);
  DECLARE_REGISTERED(dyablo::HydroUpdate_muscl_oneneighbor);
  DECLARE_REGISTERED(dyablo::HydroUpdate_euler);

  return true;
}
