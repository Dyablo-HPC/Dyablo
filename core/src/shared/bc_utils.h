/**
 * \file bc_utils.h
 * \author Pierre Kestener
 *
 * Border conditions utilities.
 */
#ifndef DYABLO_BC_UTILS_H_
#define DYABLO_BC_UTILS_H_

#include "shared/enums.h"

namespace dyablo { namespace muscl {

// =============================================================
// =============================================================
/**
 * return true if cell touches an external border.
 *
 * \param[in] dx cell size
 * \param[in] pos cell center coordinate (either x, y or z)
 *
 * \tparam bc_id an integer identifying a border (XMIN, XMAX, ...)
 */
template <int bc_id>
KOKKOS_INLINE_FUNCTION 
bool is_at_border(real_t dx, real_t pos) {

  if ((bc_id == XMIN or bc_id == YMIN or bc_id == ZMIN) and pos - dx < 0)
    return true;
  if ((bc_id == XMAX or bc_id == YMAX or bc_id == ZMAX) and pos + dx > 1.0)
    return true;

  return false;

} // is_at_border

} // namespace muscl

} // namespace dyablo

#endif // DYABLO_BC_UTILS_H_
