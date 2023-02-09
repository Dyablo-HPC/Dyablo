#pragma once

#include "gravity/GravitySolver_base.h"

namespace dyablo {

class GravitySolver_constant;
class GravitySolver_cg;

template< typename T >
class GravitySolver_analytical;
class AnalyticalFormula_gravity_point_mass;

} //namespace dyablo 


template<>
inline bool dyablo::GravitySolverFactory::init()
{
  DECLARE_REGISTERED(dyablo::GravitySolver_constant);
  DECLARE_REGISTERED(dyablo::GravitySolver_cg);
  //DECLARE_REGISTERED(dyablo::GravitySolver_analytical<dyablo::AnalyticalFormula_gravity_point_mass>);

  return true;
}

