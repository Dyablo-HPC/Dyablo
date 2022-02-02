#pragma once

#include "init/InitialConditions_base.h"

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

  return true;
}

#undef INITIALCONDITIONS_LEGACY