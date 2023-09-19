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
class AnalyticalFormula_sod;

// MHD
class AnalyticalFormula_OrszagTang;
class AnalyticalFormula_MHD_blast;
class AnalyticalFormula_MHD_rotor;

// Particles
class InitialConditions_simple_particles;
class InitialConditions_particle_grid;

//cosmo
class InitialConditions_grafic_fields;


} // namespace dyablo



template<>
bool dyablo::InitialConditionsFactory::init()
{
#ifdef DYABLO_USE_HDF5
  DECLARE_REGISTERED( dyablo::InitialConditions_restart );
#endif

  DECLARE_REGISTERED( dyablo::InitialConditions_simple_particles );
  DECLARE_REGISTERED( dyablo::InitialConditions_particle_grid );
  
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_blast> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_implode> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_riemann2d> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_KelvinHelmholtz> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_RayleighTaylor> );

  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_OrszagTang> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_blast> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_rotor> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_sod> );

  DECLARE_REGISTERED( dyablo::InitialConditions_grafic_fields );

  return true;
}

#undef INITIALCONDITIONS_LEGACY
