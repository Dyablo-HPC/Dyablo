#pragma once

#include "gravity/GravitySolver_base.h"

namespace dyablo {


class GravitySolver_constant;
class GravitySolver_cg;

} //namespace dyablo 


template<>
inline bool dyablo::GravitySolverFactory::init()
{
  DECLARE_REGISTERED(dyablo::GravitySolver_constant);
  DECLARE_REGISTERED(dyablo::GravitySolver_cg);

  return true;
}

