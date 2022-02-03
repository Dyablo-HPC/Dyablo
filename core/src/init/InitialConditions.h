#pragma once

#include "init/InitialConditions_base.h"

namespace dyablo{


template< typename AnalyticalFormula >
class InitialConditions_analytical;


class AnalyticalFormula_blast;
} // namespace dyablo



template<>
bool dyablo::InitialConditionsFactory::init()
{
  DECLARE_REGISTERED( dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_blast> );

  return true;
}

#undef INITIALCONDITIONS_LEGACY