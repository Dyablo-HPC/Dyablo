#pragma once

#include "muscl_block/gravity/GravitySolver_base.h"

namespace dyablo {
namespace muscl_block {

class GravitySolver_constant;

} //namespace dyablo 
} //namespace muscl_block

template<>
inline bool dyablo::muscl_block::GravitySolverFactory::init()
{
  DECLARE_REGISTERED(dyablo::muscl_block::GravitySolver_constant);

  return true;
}

