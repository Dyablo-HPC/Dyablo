#pragma once

#include "muscl_block/init/InitialConditions_base.h"

#define INITIALCONDITIONS_LEGACY( init_xxx, name ) \
namespace dyablo{ \
namespace muscl_block{ \
// void init_xxx(SolverHydroMusclBlock*); \
// class Init_legacy_##init_xxx : public InitialConditions{ \
//   public: \
//   virtual void init(SolverHydroMusclBlock* solver)  \
//   {  \
//     init_xxx(solver); \
//   } \
//   virtual ~Init_legacy_##init_xxx(){} \
// }; }} \
// FACTORY_REGISTER(dyablo::muscl_block::InitialConditionsFactory, dyablo::muscl_block::Init_legacy_##init_xxx, name);

// INITIALCONDITIONS_LEGACY( init_blast, "blast_legacy");
// INITIALCONDITIONS_LEGACY( init_implode, "implode");
// INITIALCONDITIONS_LEGACY( init_sod, "sod");
// INITIALCONDITIONS_LEGACY( init_kelvin_helmholtz      , "kelvin_helmholtz");
// INITIALCONDITIONS_LEGACY( init_gresho_vortex      , "gresho_vortex");
// INITIALCONDITIONS_LEGACY( init_four_quadrant      , "four_quadrant");
// INITIALCONDITIONS_LEGACY( init_isentropic_vortex, "isentropic_vortex");
// INITIALCONDITIONS_LEGACY( init_shu_osher      , "shu_osher");
// INITIALCONDITIONS_LEGACY( init_double_mach_reflection , "double_mach_reflection");
// INITIALCONDITIONS_LEGACY( init_rayleigh_taylor      , "rayleigh_taylor");

namespace dyablo{
namespace muscl_block{

template< typename AnalyticalFormula >
class InitialConditions_analytical;

} // namespace muscl_block
class AnalyticalFormula_blast;
} // namespace dyablo



template<>
bool dyablo::muscl_block::InitialConditionsFactory::init()
{
  DECLARE_REGISTERED( dyablo::muscl_block::InitialConditions_analytical<dyablo::AnalyticalFormula_blast> );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_blast );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_implode );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_sod );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_kelvin_helmholtz );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_gresho_vortex );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_four_quadrant );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_isentropic_vortex );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_shu_osher );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_double_mach_reflection );
  // DECLARE_REGISTERED( dyablo::muscl_block::Init_legacy_init_rayleigh_taylor );

  return true;
}

#undef INITIALCONDITIONS_LEGACY