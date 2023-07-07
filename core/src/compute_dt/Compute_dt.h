#pragma once

#include "compute_dt/Compute_dt_base.h"

namespace dyablo {


class Compute_dt_legacy;
class Compute_dt_hydro;
class Compute_dt_particle_velocity;

} //namespace dyablo 


template<>
inline bool dyablo::Compute_dtFactory::init()
{
  DECLARE_REGISTERED(dyablo::Compute_dt_legacy);
  DECLARE_REGISTERED(dyablo::Compute_dt_hydro);
  DECLARE_REGISTERED(dyablo::Compute_dt_particle_velocity);

  return true;
}