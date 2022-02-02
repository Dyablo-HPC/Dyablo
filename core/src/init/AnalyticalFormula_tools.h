#pragma once

#include "utils_hydro.h"

namespace dyablo {
namespace AnalyticalFormula_tools{

constexpr static real_t eps = std::numeric_limits<real_t>::epsilon();
constexpr static real_t epsref = 0.01;

/// determines if the cell at position {x,y,z} with size {dx,dy,dz} needs to be refined
template< typename Formula_t >
KOKKOS_INLINE_FUNCTION
bool auto_refine(   const Formula_t& formula,
                    real_t gamma0, real_t smallr, real_t smallp, real_t error_max,
                    real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) 
{
    real_t err = 0;
    for( ComponentIndex3D dir : {IX,IY,IZ})
    {
        HydroState3d uc = formula.value(x               ,y              ,z              ,dx,dy,dz);
        HydroState3d ul = formula.value(x-(dir==IX)*dx  ,y-(dir==IY)*dy ,z-(dir==IZ)*dz ,dx,dy,dz);
        HydroState3d ur = formula.value(x+(dir==IX)*dx  ,y+(dir==IY)*dy ,z+(dir==IZ)*dz ,dx,dy,dz);

        real_t c;

        HydroState3d Ql, Qc, Qr;
        computePrimitives( ul, &c, Ql, gamma0, smallr, smallp );
        computePrimitives( uc, &c, Qc, gamma0, smallr, smallp );
        computePrimitives( ur, &c, Qr, gamma0, smallr, smallp );

        for( VarIndex var : {ID, IP} )
        {
            real_t qm1 = Ql[var]; 
            real_t q   = Qc[var]; 
            real_t qp1 = Qr[var]; 

            real_t fr = qp1 - q;    
            real_t fl = qm1 - q;

            real_t fc = FABS(qp1) + FABS(qm1) + 2 * FABS(q);
            real_t err_local = FABS(fr + fl) / (FABS(fr) + FABS(fl) + epsref * fc + eps);

            err = fmax(err, err_local);
        }
    }

    return err > error_max;
}

} // namespace AnalyticalFormula_tools
} // namespace dyablo