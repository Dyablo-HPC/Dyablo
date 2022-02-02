#pragma once

#include "HydroUpdate_base.h"

namespace dyablo {
namespace muscl_block {

class HydroUpdate_legacy;
class HydroUpdate_generic;
class HydroUpdate_muscl_oneneighbor;

} //namespace dyablo 
} //namespace muscl_block

template<>
inline bool dyablo::muscl_block::HydroUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::muscl_block::HydroUpdate_legacy);
  DECLARE_REGISTERED(dyablo::muscl_block::HydroUpdate_generic);
  DECLARE_REGISTERED(dyablo::muscl_block::HydroUpdate_muscl_oneneighbor);

  return true;
}