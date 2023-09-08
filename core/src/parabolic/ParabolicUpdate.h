#pragma once

#include "ParabolicUpdate_base.h"

namespace dyablo {

class ParabolicUpdate_explicit;

} //namespace dyablo 

template<>
inline bool dyablo::ParabolicUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::ParabolicUpdate_explicit);

  return true;
}
