/**
 * \file HydroInitFunctor.h
 */
#ifndef HYDRO_INIT_FUNCTORS_H_
#define HYDRO_INIT_FUNCTORS_H_

namespace euler_pablo {
namespace muscl {

class SolverHydroMuscl;

// this is were actual initialization happens
void init_implode(SolverHydroMuscl *psolver);
void init_sod(SolverHydroMuscl *psolver);
void init_blast(SolverHydroMuscl *psolver);
void init_kelvin_helmholtz(SolverHydroMuscl *psolver);
void init_gresho_vortex(SolverHydroMuscl *psolver);
void init_four_quadrant(SolverHydroMuscl *psolver);
void init_isentropic_vortex(SolverHydroMuscl *psolver);
void init_rayleigh_taylor(SolverHydroMuscl *psolver);

} // namespace muscl
} // namespace euler_pablo

#endif // HYDRO_INIT_FUNCTORS_H_
