#pragma once

#include "init/InitialConditions_base.h"

namespace dyablo{


template< typename AnalyticalFormula >
class InitialConditions_analytical;

// Restart
class InitialConditions_restart;

// Hydrodynamics
class AnalyticalFormula_blast;
class AnalyticalFormula_implode;
class AnalyticalFormula_riemann2d;
class AnalyticalFormula_KelvinHelmholtz;
class AnalyticalFormula_RayleighTaylor;

// MHD
class AnalyticalFormula_OrszagTang;
class AnalyticalFormula_MHD_blast;
class AnalyticalFormula_MHD_rotor;
} // namespace dyablo



template<>
bool dyablo::InitialConditionsFactory::init()
{
  // DECLARE_REGISTERED( dyablo::InitialConditions_restart );
  
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_blast> );
  // DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_implode> );
  // DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_riemann2d> );
  // DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_KelvinHelmholtz> );
  // DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_RayleighTaylor> );

  // DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_OrszagTang> );
  // DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_blast> );
  // DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_rotor> );

  return true;
}

#undef INITIALCONDITIONS_LEGACY
