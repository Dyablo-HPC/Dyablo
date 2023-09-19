#pragma once

#include "compute_dt/Compute_dt_base.h"

namespace dyablo {


class Compute_dt_legacy;
class Compute_dt_hydro;
class Compute_dt_particle_velocity;
class Compute_dt_cosmology;

} //namespace dyablo 


template<>
inline bool dyablo::Compute_dtFactory::init()
{
  DECLARE_REGISTERED(dyablo::Compute_dt_legacy);
  DECLARE_REGISTERED(dyablo::Compute_dt_hydro);
  DECLARE_REGISTERED(dyablo::Compute_dt_particle_velocity);
  DECLARE_REGISTERED(dyablo::Compute_dt_cosmology);

  return true;
}