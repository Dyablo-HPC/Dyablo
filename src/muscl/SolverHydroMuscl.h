/**
 * \file SolverHydroMuscl.h
 * \author Pierre Kestener 
 *
 * Class SolverHydroMuscl definition.
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

private:
  //! enum use in synchronize ghost data operation to
  //! identify which variables need to be exchange by MPI
  enum class UserDataCommType {UDATA, QDATA, SLOPES};

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

  DataArray     U;     /*!< hydrodynamics conservative variables arrays at t_n */
  DataArrayHost Uhost; /*!< mirror DataArray U on host memory space */
  DataArray     U2;    /*!< hydrodynamics conservative variables arrays at t_{n+1}*/
  DataArray     Q;     /*!< hydrodynamics primitive    variables array  */

  DataArray     Ughost; /*!< ghost cell data */
  DataArray     Qghost; /*!< ghost cell data for primitive variables */

  DataArray Slopes_x; /*!< implementation 1 only */
  DataArray Slopes_y; /*!< implementation 1 only */
  DataArray Slopes_z; /*!< implementation 1 only */

  DataArray Slopes_x_ghost; /*!< implementation 1 only */
  DataArray Slopes_y_ghost; /*!< implementation 1 only */
  DataArray Slopes_z_ghost; /*!< implementation 1 only */

  /* Gravity field */
  DataArray gravity;

  //! field manager for scalar variables mapping to memory index
  FieldManager fieldMgr;
  
  /*
   * methods
   */

  void resize_solver_data();

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

  //! override base class, do_amr_cycle is supposed to be called after
  //! the numerical scheme (godunov_unsplit)
  void do_amr_cycle();
  
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

  void reconstruct_gradients(DataArray Udata);

  void compute_fluxes_and_update(DataArray data_in, DataArray data_out, real_t dt);

  //void computeTrace(DataArray Udata, real_t dt);
  
  // output
  void save_solution_impl();

private:

  /*
   * the following routines are necessary for amr cycle.
   */
  
  //! synchonize ghost data / only necessary when MPI is activated
  void synchronize_ghost_data(UserDataCommType t);

  //! mark cells for refinement
  void mark_cells();

  //! adapt mesh and recompute connectivity
  void adapt_mesh();

  //! map data from old U to new U after adapting mesh
  void map_userdata_after_adapt();

  //! mesh load balancing with data communication
  void load_balance_userdata();
  
}; // class SolverHydroMuscl

} // namespace muscl

} // namespace euler_pablo

#endif // SOLVER_HYDRO_MUSCL_H_
