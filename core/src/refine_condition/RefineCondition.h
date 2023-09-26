#pragma once

#include "refine_condition/RefineCondition_base.h"

namespace dyablo {


class RefineCondition_legacy;
class RefineCondition_second_derivative_error;
class RefineCondition_pseudo_gradient;
class RefineCondition_mass;
class RefineCondition_downflows;

} //namespace dyablo 


template<>
inline bool dyablo::RefineConditionFactory::init()
{
  DECLARE_REGISTERED(dyablo::RefineCondition_legacy);
  DECLARE_REGISTERED(dyablo::RefineCondition_second_derivative_error);
  DECLARE_REGISTERED(dyablo::RefineCondition_pseudo_gradient);
  DECLARE_REGISTERED(dyablo::RefineCondition_mass);
  DECLARE_REGISTERED(dyablo::RefineCondition_downflows);

  return true;
}