#pragma once

#include "gravity/GravitySolver_base.h"

namespace dyablo {


class GravitySolver_constant;

} //namespace dyablo 


template<>
inline bool dyablo::GravitySolverFactory::init()
{
  DECLARE_REGISTERED(dyablo::GravitySolver_constant);

  return true;
}

