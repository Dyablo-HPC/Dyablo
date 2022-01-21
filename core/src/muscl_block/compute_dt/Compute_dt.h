#pragma once

#include "muscl_block/compute_dt/Compute_dt_base.h"

namespace dyablo {
namespace muscl_block {

class Compute_dt_legacy;

} //namespace dyablo 
} //namespace muscl_block

template<>
inline bool dyablo::muscl_block::Compute_dtFactory::init()
{
  DECLARE_REGISTERED(dyablo::muscl_block::Compute_dt_legacy);

  return true;
}