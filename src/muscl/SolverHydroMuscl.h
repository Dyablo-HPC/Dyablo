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
#include "shared/FieldManager.h"

// for IO
#include <shared/HDF5_IO.h>

namespace dyablo { namespace muscl {

/**
 * Main hydrodynamics data structure for 2D/3D MUSCL-Hancock scheme.
 */
class SolverHydroMuscl : public dyablo::SolverBase
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

  //! resize all workspace data array
  void resize_solver_data();

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

  //! numerical scheme - wrapper around godunov_unsplit_impl 
  void godunov_unsplit(real_t dt);

  //! numerical scheme
  void godunov_unsplit_impl(DataArray data_in, 
			    DataArray data_out, 
			    real_t dt);

  //! convert conservative variables to primitive variables
  void convertToPrimitives(DataArray Udata);

  //! reconstruct gradients / limited slopes
  void reconstruct_gradients(DataArray Udata);

  //! compute flux (Riemann solver) and perform time update
  void compute_fluxes_and_update(DataArray data_in, DataArray data_out, real_t dt);

  //! output
  void save_solution_impl();

private:

#ifdef USE_HDF5
  std::shared_ptr<HDF5_Writer> hdf5_writer;
#endif // USE_HDF5

  //! VTK output
  void save_solution_vtk();
  
  //! HDF5 output
  void save_solution_hdf5();

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

} // namespace dyablo

#endif // SOLVER_HYDRO_MUSCL_H_
