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
  
  void computeFluxesAndUpdate(DataArray Udata, 
			      real_t dt);

  // output
  void save_solution_impl();
  
  int isize, jsize, ksize;
  int nbCells;
  
}; // class SolverHydroMuscl

// =======================================================
// ==== CLASS SolverHydroMuscl IMPL ======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl::SolverHydroMuscl(HydroParams& params,
				   ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  Slopes_x(), Slopes_y(), Slopes_z()
{
  
  solver_type = SOLVER_MUSCL_HANCOCK;
  
  m_nCells = nbCells;
  m_nDofsPerCell = 1;

  int nbvar = params.nbvar;
 
  long long int total_mem_size = 0;

  /*
   * memory pre-allocation.
   *
   * Note that Uhost is not just a view to U, Uhost will be used
   * to save data from multiple other device array.
   * That's why we didn't use create_mirror_view to initialize Uhost.
   */

  // minimal number of cells
  uint64_t nbCells = 2; //1<<params.level_min;
  
  nbCells = params.dimType == TWO_D ? nbCells * nbCells :  nbCells * nbCells * nbCells;
  
  U     = DataArray("U", nbCells, nbvar);
  Uhost = Kokkos::create_mirror(U);
  U2    = DataArray("U2",nbCells, nbvar);
  Q     = DataArray("Q", nbCells, nbvar);
  
  total_mem_size += nbCells*nbvar * sizeof(real_t) * 3;// 1+1+1 for U+U2+Q
  
  if (params.implementationVersion == 0) {
      
    Fluxes_x = DataArray("Fluxes_x", nbCells, nbvar);
    Fluxes_y = DataArray("Fluxes_y", nbCells, nbvar);

    if (m_dim == 3)
      Fluxes_z = DataArray("Fluxes_z", isize,jsize,ksize, nbvar);

    if (m_dim == 2)
      total_mem_size += nbCells*nbvar * sizeof(real_t) * 2;// 1+1 for Fluxes_x+Fluxes_y
    else
      total_mem_size += nbCells*nbvar * sizeof(real_t) * 3;// 1+1+1 for Fluxes_x+Fluxes_y+Fluxes_z
    
  } else if (params.implementationVersion == 1) {
      
    Slopes_x = DataArray("Slope_x", nbCells, nbvar);
    Slopes_y = DataArray("Slope_y", nbCells, nbvar);

    if (m_dim == 3)
      Slopes_z = DataArray("Slope_z", nbCells, nbvar);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray("Fluxes_x", nbCells, nbvar);
    Fluxes_y = Fluxes_x;
    if (m_dim == 3)
      Fluxes_z = Fluxes_x;

    if (m_dim==2)
      total_mem_size += nbCells*nbvar * sizeof(real_t) * 3;// 1+1+1 for Slopes_x+Slopes_y+Fluxes_x
    else
      total_mem_size += nbCells*nbvar * sizeof(real_t) * 4;// 1+1+1 for Slopes_x+Slopes_y+Slopes_z+Fluxes_x
    
  } 
  
  // if (m_gravity_enabled) {
  //   gravity = DataArray("gravity field",nbCells,m_dim);
  //   total_mem_size += isize*jsize*2; // TODO
  // }      
 
  // perform init condition
  init(U);
  
  // initialize boundaries
  //make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2,U);
  
  // compute initialize time step
  compute_dt();

  int myRank=0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI

  if (myRank==0) {
    std::cout << "##########################" << "\n";
    std::cout << "Solver is " << m_solver_name << "\n";
    std::cout << "Problem (init condition) is " << m_problem_name << "\n";
    std::cout << "##########################" << "\n";
    
    // print parameters on screen
    params.print();
    std::cout << "##########################" << "\n";
    std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes\n"; 
    std::cout << "##########################" << "\n";
  }

} // SolverHydroMuscl::SolverHydroMuscl

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl::~SolverHydroMuscl()
{

} // SolverHydroMuscl::~SolverHydroMuscl


// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMuscl::compute_dt_local()
{


} // SolverHydroMuscl::compute_dt_local

// =======================================================
// =======================================================
void SolverHydroMuscl::next_iteration_impl()
{
  
  int myRank=0;
  
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI
  
  if (m_iteration % m_nlog == 0) {
    if (myRank==0) {
      printf("time step=%7d (dt=% 10.8f t=% 10.8f)\n",m_iteration,m_dt, m_t);
    }
  }
  
  // output
  if (params.enableOutput) {
    if ( should_save_solution() ) {
      
      if (myRank==0) {
	std::cout << "Output results at time t=" << m_t
		  << " step " << m_iteration
		  << " dt=" << m_dt << std::endl;
      }
      
      save_solution();
      
    } // end output
  } // end enable output
  
  // compute new dt
  m_timers[TIMER_DT]->start();
  compute_dt();
  m_timers[TIMER_DT]->stop();
  
  // perform one step integration
  godunov_unsplit(m_dt);
  
} // SolverHydroMuscl::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverHydroMuscl::godunov_unsplit(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    godunov_unsplit_impl(U , U2, dt);
  } else {
    godunov_unsplit_impl(U2, U , dt);
  }
  
} // SolverHydroMuscl::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverHydroMuscl::convertToPrimitives(DataArray Udata)
{

  
} // SolverHydroMuscl::convertToPrimitives

// =======================================================
// =======================================================
void SolverHydroMuscl::save_solution_impl()
{

  m_timers[TIMER_IO]->start();
  if (m_iteration % 2 == 0)
    save_data(U,  Uhost, m_times_saved, m_t);
  else
    save_data(U2, Uhost, m_times_saved, m_t);
  
  m_timers[TIMER_IO]->stop();
    
} // SolverHydroMuscl::save_solution_impl()

} // namespace muscl

} // namespace euler_pablo

#endif // SOLVER_HYDRO_MUSCL_H_
