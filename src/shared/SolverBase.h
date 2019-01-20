#ifndef SOLVER_BASE_H_
#define SOLVER_BASE_H_

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"
#include "shared/kokkos_shared.h"

#include <map>
#include <memory> // for std::unique_ptr / std::shared_ptr

// for timer
#ifdef KOKKOS_ENABLE_CUDA
#include "utils/monitoring/CudaTimer.h"
#else
#include "utils/monitoring/OpenMPTimer.h"
#endif

//! this enum helps identifying the type of solver used
enum solver_type_t {
  SOLVER_UNDEFINED=0,
  SOLVER_MUSCL_HANCOCK=1
};

namespace euler_pablo { namespace io {
class IO_ReadWriteBase;
} }

enum TimerIds {
  TIMER_TOTAL = 0,
  TIMER_IO = 1,
  TIMER_DT = 2,
  TIMER_BOUNDARIES = 3,
  TIMER_NUM_SCHEME = 4
}; // enum TimerIds

namespace euler_pablo {

/**
 * Abstract base class for all our actual solvers.
 */
class SolverBase {
  
public:
  
  SolverBase(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverBase();

  // hydroParams
  HydroParams& params;
  ConfigMap& configMap;

  /* some common member data */
  solver_type_t solver_type;
  
  //! is this a restart run ?
  int m_restart_run_enabled;

  //! filename containing data from a previous run.
  std::string m_restart_run_filename;
  
  // iteration info
  double               m_t;         //!< the time at the current iteration
  double               m_dt;        //!< the time step at the current iteration
  int                  m_iteration; //!< the current iteration (integer)
  double               m_tEnd;      //!< maximun time
  double               m_cfl;       //!< Courant number
  int                  m_nlog;      //!< number of steps between two monitoring print on screen
  
  long long int        m_nCells;       //!< number of cells
  long long int        m_nDofsPerCell; //!< number of degrees of freedom per cell 

  //! init condition name (or problem)
  std::string          m_problem_name;
  
  //! solver name (use in output file).
  std::string          m_solver_name;

  //! dimension (2 or 3)
  int                  m_dim;
  
  /*
   *
   * Computation interface that may be overriden in a derived 
   * concrete implementation.
   *
   */

  //! Read and parse the configuration file (ini format).
  virtual void read_config();

  //! Compute CFL condition (allowed time step), over all MPI process.
  virtual void compute_dt();

  //! Compute CFL condition local to current MPI process
  virtual double compute_dt_local();

  //! Check if current time is larger than end time.
  virtual int finished();

  //! This is where action takes place. Wrapper arround next_iteration_impl.
  virtual void next_iteration();

  //! This is the next iteration computation (application specific).
  virtual void next_iteration_impl();
  
  //! Decides if the current time step is eligible for dump data to file
  virtual int  should_save_solution();

  //! main routine to dump solution to file
  virtual void save_solution();

  //! main routine to dump solution to file
  virtual void save_solution_impl();

  //! read restart data
  virtual void read_restart_file();
  
  /* IO related */

  //! counter incremented each time an output is written
  int m_times_saved;

  //! Number of variables to saved
  //int m_write_variables_number;

  //! names of variables to save
  std::map<int, std::string> m_variables_names;

  //! timers
#ifdef KOKKOS_ENABLE_CUDA
  using Timer = CudaTimer;
#else
  using Timer = OpenMPTimer;
#endif
  using TimerMap = std::map<int, std::shared_ptr<Timer> >;
  TimerMap m_timers;

  void save_data(DataArray             U,
		 DataArray::HostMirror Uh,
		 int iStep,
		 real_t time);

  void save_data_debug(DataArray             U,
		       DataArray::HostMirror Uh,
		       int iStep,
		       real_t time,
		       std::string debug_name);

  /** 
   * Routine to load data from file (for a restart run). 
   * This routine change iStep and time (loaded from file).
   */
  void load_data(DataArray             U,
		 DataArray::HostMirror Uh,
		 int& iStep,
		 real_t& time);
  
  
  virtual void make_boundary(DataArray Udata, FaceIdType faceId, bool mhd_enabled);
  virtual void make_boundaries_serial(DataArray Udata, bool mhd_enabled);

#ifdef USE_MPI
  virtual void make_boundaries_mpi(DataArray Udata, bool mhd_enabled);
#endif // USE_MPI

}; // class SolverBase

} // namespace euler_pablo

#endif // SOLVER_BASE_H_
