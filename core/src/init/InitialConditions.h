#pragma once

#include "init/InitialConditions_base.h"

namespace dyablo{


template< typename AnalyticalFormula >
class InitialConditions_analytical;


class AnalyticalFormula_blast;
class AnalyticalFormula_implode;
class AnalyticalFormula_riemann2d;
class AnalyticalFormula_KH;
} // namespace dyablo



template<>
bool dyablo::InitialConditionsFactory::init()
{
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_blast> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_implode> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_riemann2d> );
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_KH> );

  return true;
}

#undef INITIALCONDITIONS_LEGACY
