#pragma once

#include "utils_hydro.h"

namespace dyablo {

class AnalyticalFormula_base{
public:

    /// determines if the cell at position {x,y,z} with size {dx,dy,dz} needs to be refined
    KOKKOS_INLINE_FUNCTION
    bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const;
    /// the final value hydro state for the cell at position {x,y,z} with size {dx,dy,dz}
    KOKKOS_INLINE_FUNCTION
    ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const;
};

} // namespace dyablo