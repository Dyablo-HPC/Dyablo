#pragma once

#include "MusclBlockUpdate_base.h"

namespace dyablo {
namespace muscl_block {

class MusclBlockUpdate_legacy;
class MusclBlockUpdate_generic;

} //namespace dyablo 
} //namespace muscl_block

template<>
inline bool dyablo::muscl_block::MusclBlockUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::muscl_block::MusclBlockUpdate_legacy);
  DECLARE_REGISTERED(dyablo::muscl_block::MusclBlockUpdate_generic);

  return true;
}