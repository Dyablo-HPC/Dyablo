#ifndef HYDRO_INIT_FUNCTORS_H_
#define HYDRO_INIT_FUNCTORS_H_

namespace euler_pablo {
namespace muscl {

class SolverHydroMuscl;
void init_four_quadrant(SolverHydroMuscl *psolver);
void init_sod(SolverHydroMuscl *psolver);

} // namespace muscl
} // namespace euler_pablo

#include "InitBlast.h"
#include "InitIsentropicVortex.h"
#include "InitKelvinHelmholtz.h"

#endif // HYDRO_INIT_FUNCTORS_H_
