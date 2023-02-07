#pragma once

#include "HydroUpdate_base.h"
#include "states/State_forward.h"

namespace dyablo {


class HydroUpdate_legacy;

template<typename State> class HydroUpdate_hancock_oneneighbor;
template<typename State> class HydroUpdate_hancock;
template<typename State> class HydroUpdate_euler;
template<typename State> class HydroUpdate_RK2;

} //namespace dyablo 


template<>
inline bool dyablo::HydroUpdateFactory::init()
{
  // DECLARE_REGISTERED(dyablo::HydroUpdate_legacy);
  
  // DECLARE_REGISTERED(dyablo::HydroUpdate_hancock_oneneighbor<dyablo::HydroState>);
  // DECLARE_REGISTERED(dyablo::HydroUpdate_hancock_oneneighbor<dyablo::MHDState>);
  DECLARE_REGISTERED(dyablo::HydroUpdate_hancock<dyablo::HydroState>);
  // DECLARE_REGISTERED(dyablo::HydroUpdate_hancock<dyablo::MHDState>);
  // DECLARE_REGISTERED(dyablo::HydroUpdate_euler<dyablo::HydroState>);
  // DECLARE_REGISTERED(dyablo::HydroUpdate_euler<dyablo::MHDState>);
  // DECLARE_REGISTERED(dyablo::HydroUpdate_RK2<dyablo::HydroState>);
  // DECLARE_REGISTERED(dyablo::HydroUpdate_RK2<dyablo::MHDState>);

  return true;
}
