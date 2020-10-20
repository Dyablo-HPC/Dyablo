#ifndef DYABLO_SHARED_POINT_H
#define DYABLO_SHARED_POINT_H

#include <array>
#include "shared/real_type.h"

namespace dyablo
{

/** typedef Point holding coordinates of a point. */
template<int dim>
using Point = std::array<real_t, dim>;

} // namespace dyablo

#endif // DYABLO_SHARED_POINT_H
