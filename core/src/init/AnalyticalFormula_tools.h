#pragma once

#include "utils_hydro.h"

namespace dyablo {
namespace AnalyticalFormula_tools{

constexpr static real_t eps = std::numeric_limits<real_t>::epsilon();
constexpr static real_t epsref = 0.01;

/// determines if the cell at position {x,y,z} with size {dx,dy,dz} needs to be refined
template< 
  typename Formula_t
   >
KOKKOS_INLINE_FUNCTION
bool auto_refine( const Formula_t& formula,
                  real_t gamma0, real_t smallr, real_t smallp, real_t error_max,
                  real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) 
{
    real_t err = 0;
    for( ComponentIndex3D dir : {IX,IY,IZ})
    {
        auto uc = formula.value(x               ,y              ,z              ,dx,dy,dz);
        auto ul = formula.value(x-(dir==IX)*dx  ,y-(dir==IY)*dy ,z-(dir==IZ)*dz ,dx,dy,dz);
        auto ur = formula.value(x+(dir==IX)*dx  ,y+(dir==IY)*dy ,z+(dir==IZ)*dz ,dx,dy,dz);

        auto Ql = consToPrim<3>(ul, gamma0);
        auto Qc = consToPrim<3>(uc, gamma0);
        auto Qr = consToPrim<3>(ur, gamma0);

        auto compute_err = [&](real_t qm1, real_t q, real_t qp1) {
            real_t fr = qp1 - q;    
            real_t fl = qm1 - q;

            real_t fc = FABS(qp1) + FABS(qm1) + 2 * FABS(q);
            return FABS(fr + fl) / (FABS(fr) + FABS(fl) + epsref * fc + eps);
        };

        err = fmax(err, compute_err(Ql.rho, Qc.rho, Qr.rho));
        err = fmax(err, compute_err(Ql.p,   Qc.p,   Qr.p));
    }

    return err > error_max;
}

} // namespace AnalyticalFormula_tools
} // namespace dyablo