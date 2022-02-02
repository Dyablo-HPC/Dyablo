#pragma once

#include "refine_condition/RefineCondition_base.h"

namespace dyablo {
namespace muscl_block {

class RefineCondition_legacy;
class RefineCondition_second_derivative_error;
class RefineCondition_pseudo_gradient;

} //namespace dyablo 
} //namespace muscl_block

template<>
inline bool dyablo::muscl_block::RefineConditionFactory::init()
{
  DECLARE_REGISTERED(dyablo::muscl_block::RefineCondition_legacy);
  DECLARE_REGISTERED(dyablo::muscl_block::RefineCondition_second_derivative_error);
  DECLARE_REGISTERED(dyablo::muscl_block::RefineCondition_pseudo_gradient);

  return true;
}