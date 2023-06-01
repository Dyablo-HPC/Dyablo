#pragma once

#include "ParticleUpdate_base.h"

namespace dyablo {


class ParticleUpdate_tracers_move;

} //namespace dyablo 


template<>
inline bool dyablo::ParticleUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::ParticleUpdate_tracers_move);

  return true;
}
