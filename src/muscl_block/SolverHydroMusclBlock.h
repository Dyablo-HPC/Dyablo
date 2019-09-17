/**
 * \file SolverHydroMusclBlock.h
 * \author Pierre Kestener 
 *
 * Class SolverHydroMusclBlock definition.
 *
 * Main class for solving hydrodynamics (Euler) with
 * MUSCL-Hancock scheme for 2D/3D.
 */
#ifndef SOLVER_HYDRO_MUSCL_BLOCK_H_
#define SOLVER_HYDRO_MUSCL_BLOCK_H_

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
#include "muscl_block/utils_block.h"

// for IO
#include <shared/HDF5_IO.h>

namespace dyablo { namespace muscl_block {

/**
 * Main hydrodynamics data structure for 2D/3D MUSCL-Hancock scheme.
 */
class SolverHydroMusclBlock : public dyablo::SolverBase
{

private:
  //! enum use in synchronize ghost data operation to
  //! identify which variables need to be exchange by MPI
  enum class UserDataCommType {UDATA, QDATA, SLOPES};

public:

  SolverHydroMusclBlock(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverHydroMusclBlock();
  
  /**
   * Static creation method called by the solver factory.
   */
  static SolverBase* create(HydroParams& params, ConfigMap& configMap)
  {
    SolverHydroMusclBlock* solver = new SolverHydroMusclBlock(params, configMap);

    return solver;
  }

  DataArrayBlock     U;     /*!< hydrodynamics conservative variables arrays at t_n */
  DataArrayBlockHost Uhost; /*!< mirror DataArrayBlock U on host memory space */
  DataArrayBlock     U2;    /*!< hydrodynamics conservative variables arrays at t_{n+1}*/
  DataArrayBlock     Q;     /*!< hydrodynamics primitive    variables array  */

  DataArrayBlock     Ughost; /*!< ghost cell data */
  DataArrayBlock     Qghost; /*!< ghost cell data for primitive variables */

  DataArrayBlock Slopes_x; /*!< implementation 1 only */
  DataArrayBlock Slopes_y; /*!< implementation 1 only */
  DataArrayBlock Slopes_z; /*!< implementation 1 only */

  DataArrayBlock Slopes_x_ghost; /*!< implementation 1 only */
  DataArrayBlock Slopes_y_ghost; /*!< implementation 1 only */
  DataArrayBlock Slopes_z_ghost; /*!< implementation 1 only */

  // fluxes - only usefull when low Mach RSST computation is activated
  DataArrayBlock Fluxes;

  // Gravity field
  DataArrayBlock gravity;

  //! field manager for scalar variables mapping to memory index
  FieldManager fieldMgr;
  
  /*
   * methods
   */

  //! resize all workspace data array
  void resize_solver_data();

  //! init restart (load data from file)
  void init_restart(DataArrayBlock Udata);
  
  //! init wrapper (actual initialization)
  void init(DataArrayBlock Udata);

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
  void godunov_unsplit_impl(DataArrayBlock data_in, 
			    DataArrayBlock data_out, 
			    real_t dt);

  //! convert conservative variables to primitive variables
  void convertToPrimitives(DataArrayBlock Udata);

  //! reconstruct gradients / limited slopes
  void reconstruct_gradients(DataArrayBlock Udata);

  //! compute flux (Riemann solver) and perform time update
  void compute_fluxes_and_update(DataArrayBlock data_in, 
                                 DataArrayBlock data_out,
                                 real_t dt);

  //! output
  void save_solution_impl();

  //! block sizes
  uint32_t bx, by, bz;

  //! block sizes as an array
  blockSize_t blockSizes;

  //! number of cells per octant
  uint32_t nbCellsPerOct;

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

}; // class SolverHydroMusclBlock

} // namespace muscl_block

} // namespace dyablo

#endif // SOLVER_HYDRO_MUSCL_BLOCK_H_
