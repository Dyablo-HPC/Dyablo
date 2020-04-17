#ifndef KHAMR_POINT_H
#define KHAMR_POINT_H

#include <array>
#include "shared/real_type.h"

namespace dyablo {

/** typedef Point holding coordinates of a point. */
template<int dim>
using Point = std::array<real_t, dim>;

} // namespace

#endif // DYABLO_POINT_H
