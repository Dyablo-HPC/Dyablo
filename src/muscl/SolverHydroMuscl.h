/**
 * Class SolverHydroMuscl implementation.
 *
 * Main class for solving hydrodynamics (Euler) with
 * MUSCL-Hancock scheme for 2D/3D.
 */
#ifndef SOLVER_HYDRO_MUSCL_H_
#define SOLVER_HYDRO_MUSCL_H_

#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

// shared
#include "shared/SolverBase.h"
#include "shared/HydroParams.h"
#include "shared/kokkos_shared.h"

// the actual computational functors called in HydroRun
//#include "muscl/HydroRunFunctors2D.h"
//#include "muscl/HydroRunFunctors3D.h"

// Init conditions functors
//#include "muscl/HydroInitFunctors2D.h"
//#include "muscl/HydroInitFunctors3D.h"

// border conditions functors
//#include "shared/BoundariesFunctors.h"

// for IO
//#include <utils/io/IO_ReadWrite.h>

// for init condition
#include "shared/problems/initRiemannConfig2d.h"
// #include "shared/problems/BlastParams.h"
// #include "shared/problems/ImplodeParams.h"
// #include "shared/problems/KHParams.h"
// #include "shared/problems/GreshoParams.h"
// #include "shared/problems/IsentropicVortexParams.h"

namespace euler_pablo { namespace muscl {

/**
 * Main hydrodynamics data structure for 2D/3D MUSCL-Hancock scheme.
 */
class SolverHydroMuscl : public euler_pablo::SolverBase
{

public:

  SolverHydroMuscl(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverHydroMuscl();
  
  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverHydroMuscl* solver = new SolverHydroMuscl(params, configMap);

    return solver;
  }

  DataArray     U;     /*!< hydrodynamics conservative variables arrays */
  DataArrayHost Uhost; /*!< U mirror on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays */
  DataArray     Q;     /*!< hydrodynamics primitive    variables array  */

  /* implementation 0 */
  DataArray Fluxes_x; /*!< implementation 0 */
  DataArray Fluxes_y; /*!< implementation 0 */
  DataArray Fluxes_z; /*!< implementation 0 */
  
  /* implementation 1 only */
  DataArray Slopes_x; /*!< implementation 1 only */
  DataArray Slopes_y; /*!< implementation 1 only */
  DataArray Slopes_z; /*!< implementation 1 only */


  /* Gravity field */
  DataArray gravity;
  
  //riemann_solver_t riemann_solver_fn; /*!< riemann solver function pointer */

  /*
   * methods
   */

  // fill boundaries / ghost 2d / 3d
  void make_boundaries(DataArray Udata);

  // host routines (initialization)  
  void init_implode(DataArray Udata); // 2d and 3d
  void init_blast(DataArray Udata); // 2d and 3d
  void init_kelvin_helmholtz(DataArray Udata); // 2d and 3d
  void init_gresho_vortex(DataArray Udata); // 2d and 3d
  void init_four_quadrant(DataArray Udata); // 2d only
  void init_isentropic_vortex(DataArray Udata); // 2d only
  void init_rayleigh_taylor(DataArray Udata, DataArray gravity); // 2d and 3d
  void init_rising_bubble(DataArray Udata, DataArray gravity); // 2d and 3d
  void init_disk(DataArray Udata, DataArray gravity); // 2d and 3d

  //! init restart (load data from file)
  void init_restart(DataArray Udata);
  
  //! init wrapper (actual initialization)
  void init(DataArray Udata);

  //! compute time step inside an MPI process, at shared memory level.
  double compute_dt_local();

  //! perform 1 time step (time integration).
  void next_iteration_impl();

  //! numerical scheme
  void godunov_unsplit(real_t dt);
  
  void godunov_unsplit_impl(DataArray data_in, 
			    DataArray data_out, 
			    real_t dt);
  
  void convertToPrimitives(DataArray Udata);
  
  //void computeTrace(DataArray Udata, real_t dt);
  
  // output
  void save_solution_impl();
  
}; // class SolverHydroMuscl

} // namespace muscl

} // namespace euler_pablo

#endif // SOLVER_HYDRO_MUSCL_H_