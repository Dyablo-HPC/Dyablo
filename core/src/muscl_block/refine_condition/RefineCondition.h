#pragma once

#include "muscl_block/refine_condition/RefineCondition_base.h"

namespace dyablo {
namespace muscl_block {

class RefineCondition_legacy;

} //namespace dyablo 
} //namespace muscl_block

template<>
inline bool dyablo::muscl_block::RefineConditionFactory::init()
{
  DECLARE_REGISTERED(dyablo::muscl_block::RefineCondition_legacy);

  return true;
}