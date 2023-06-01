#pragma once

#include "ParticleUpdate_base.h"

namespace dyablo {


class ParticleUpdate_tracers_move;
class ParticleUpdate_NGP_move;
class ParticleUpdate_NGP_density;

} //namespace dyablo 


template<>
inline bool dyablo::ParticleUpdateFactory::init()
{
  DECLARE_REGISTERED(dyablo::ParticleUpdate_tracers_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_NGP_move);
  DECLARE_REGISTERED(dyablo::ParticleUpdate_NGP_density);

  return true;
}
