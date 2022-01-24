#pragma once

#include "muscl_block/refine_condition/RefineCondition_base.h"

namespace dyablo {
namespace muscl_block {

class RefineCondition_legacy;
class RefineCondition_generic;

} //namespace dyablo 
} //namespace muscl_block

template<>
inline bool dyablo::muscl_block::RefineConditionFactory::init()
{
  DECLARE_REGISTERED(dyablo::muscl_block::RefineCondition_legacy);
  DECLARE_REGISTERED(dyablo::muscl_block::RefineCondition_generic);

  return true;
}