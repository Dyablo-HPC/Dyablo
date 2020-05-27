/**
 * \file HydroInitFunctor.h
 */
#ifndef HYDRO_INIT_MUSCL_BLOCK_FUNCTORS_H_
#define HYDRO_INIT_MUSCL_BLOCK_FUNCTORS_H_

namespace dyablo {
namespace muscl_block {

class SolverHydroMusclBlock;

// this is were actual initialization happens
void init_implode(SolverHydroMusclBlock *psolver);
void init_sod(SolverHydroMusclBlock *psolver);
void init_blast(SolverHydroMusclBlock *psolver);
void init_kelvin_helmholtz(SolverHydroMusclBlock *psolver);
void init_gresho_vortex(SolverHydroMusclBlock *psolver);
void init_four_quadrant(SolverHydroMusclBlock *psolver);
void init_isentropic_vortex(SolverHydroMusclBlock *psolver);
void init_shu_osher(SolverHydroMusclBlock *psolver);
void init_double_mach_reflection(SolverHydroMusclBlock *psolver);
void init_rayleigh_taylor(SolverHydroMusclBlock *psolver);

} // namespace muscl_block
} // namespace dyablo

#endif // HYDRO_INIT_MUSCL_BLOCK_FUNCTORS_H_
