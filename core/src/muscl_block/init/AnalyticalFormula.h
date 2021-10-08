#pragma once

#include "shared/HydroState.h"

namespace dyablo {

class AnalyticalFormula_base{
public:
    /// determines if the cell at position {x,y,z} with size {dx,dy,dz} needs to be refined
    bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz );
    /// the final value hydro state for the cell at position {x,y,z} with size {dx,dy,dz}
    HydroState3d value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz );
};

} // namespace dyablo