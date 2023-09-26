#pragma once

#include "init/InitialConditions_base.h"
#include "states/State_forward.h"

namespace dyablo{


template< typename AnalyticalFormula >
class InitialConditions_analytical;

// Restart
class InitialConditions_restart;
class InitialConditions_tiled_restart;

// Hydrodynamics
class AnalyticalFormula_blast;
class AnalyticalFormula_implode;
class AnalyticalFormula_riemann2d;
class AnalyticalFormula_KelvinHelmholtz;
class AnalyticalFormula_RayleighTaylor;
class AnalyticalFormula_sod;

// MHD
template<typename State> class AnalyticalFormula_OrszagTang;
template<typename State> class AnalyticalFormula_MHD_blast;
template<typename State> class AnalyticalFormula_MHD_rotor;

// Particles
class InitialConditions_simple_particles;
class InitialConditions_particle_grid;

// Cosmology
class InitialConditions_grafic_fields;

// Convection
class AnalyticalFormula_C91;
class AnalyticalFormula_tri_layer;


} // namespace dyablo



template<>
bool dyablo::InitialConditionsFactory::init()
{
#ifdef DYABLO_USE_HDF5
  DECLARE_REGISTERED( dyablo::InitialConditions_restart );
  DECLARE_REGISTERED( dyablo::InitialConditions_tiled_restart );
#endif

  DECLARE_REGISTERED( dyablo::InitialConditions_simple_particles );
  DECLARE_REGISTERED( dyablo::InitialConditions_particle_grid );
  
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_blast> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_implode> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_riemann2d> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_KelvinHelmholtz> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_RayleighTaylor> );

  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_OrszagTang<dyablo::MHDState> > );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_OrszagTang<dyablo::GLMMHDState> > );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_blast<dyablo::MHDState>> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_blast<dyablo::GLMMHDState>> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_rotor<dyablo::MHDState>> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_rotor<dyablo::GLMMHDState>> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_sod> );

  DECLARE_REGISTERED( dyablo::InitialConditions_grafic_fields );

  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_C91> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_tri_layer> );

  return true;
}

#undef INITIALCONDITIONS_LEGACY
