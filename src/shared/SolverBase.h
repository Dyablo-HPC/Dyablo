#ifndef SOLVER_BASE_H_
#define SOLVER_BASE_H_

#include <map>
#include <memory> // for std::unique_ptr / std::shared_ptr

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"
#include "shared/kokkos_shared.h"

#ifdef USE_HDF5
#include "shared/HDF5_IO.h"
#endif // USE_HDF5

// for timer
#ifdef KOKKOS_ENABLE_CUDA
#include "utils/monitoring/CudaTimer.h"
#else
#include "utils/monitoring/OpenMPTimer.h"
#endif

#include "shared/bitpit_common.h"

//! this enum helps identifying the type of solver used
enum solver_type_t {
  SOLVER_UNDEFINED=0,
  SOLVER_MUSCL_HANCOCK=1,
  SOLVER_MUSCL_HANCOCK_BLOCK=2
};

//! enum for identifying timers
enum TimerIds {
  TIMER_TOTAL = 0,
  TIMER_IO = 1,
  TIMER_DT = 2,
  TIMER_BOUNDARIES = 3,
  TIMER_NUM_SCHEME = 4,
  TIMER_AMR_CYCLE = 5,
  TIMER_AMR_CYCLE_SYNC_GHOST = 6,
  TIMER_AMR_CYCLE_MARK_CELLS = 7,
  TIMER_AMR_CYCLE_ADAPT_MESH = 8,
  TIMER_AMR_CYCLE_MAP_USERDATA = 9,
  TIMER_AMR_CYCLE_LOAD_BALANCE = 10,
  TIMER_AMR_BLOCK_COPY = 11,
  TIMER_BLOCK_COPY = 12,
  TIMER_NB = 13 // number of TIMERs, do not remove
}; // enum TimerIds

namespace dyablo {

/**
 * Abstract base class for all our actual solvers.
 */
class SolverBase {
  
public:
  
  SolverBase(HydroParams& params, ConfigMap& configMap);
  virtual ~SolverBase();

  //! hydrodynamics parameters settings
  HydroParams& params;

  //! unordered map of parameters read from input ini file
  ConfigMap& configMap;

  //! The main AMR object (from bitpit library)
  std::shared_ptr<AMRmesh> amr_mesh;

  //! number of children: 4 in 2D, 8 in 3D
  uint8_t m_nbChildren;
  
  //! enum type to the actual solver type (Hydro, MHD, ...). TBC if needed.
  solver_type_t solver_type;
  
  //! is this a restart run ?
  int m_restart_run_enabled;

  //! filename containing data from a previous run.
  std::string m_restart_run_filename;
  
  // iteration info
  double               m_t;         //!< the time at the current iteration
  double               m_dt;        //!< the time step at the current iteration
  int                  m_iteration; //!< the current iteration (integer)
  int                  m_max_iterations; //!< user defined maximum iteration count
  double               m_tBeg;      //!< begin time (might be non-zero upon restart)
  double               m_tEnd;      //!< maximun time
  double               m_cfl;       //!< Courant number
  int                  m_nlog;      //!< number of steps between two monitoring print on screen

  int                  m_max_save_count; //!< max number of output written
  int                  m_max_restart_count; //!< max number of restart file written

  
  long long int        m_nCells;       //!< number of cells
  long long int        m_nDofsPerCell; //!< number of degrees of freedom per cell 

  // statistics
  //! total number of quadrant update
  uint64_t             m_total_num_cell_updates;
  
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

  //! perform AMR cycle (mark cells, adapt = refine/coarsen, remap user data)
  virtual void do_amr_cycle();

  //! do MPI load balancing
  virtual void do_load_balancing();

  //! Read and parse the configuration file (ini format).
  virtual void read_config();

  //! Compute CFL condition (allowed time step), over all MPI process.
  virtual void compute_dt();

  //! Compute CFL condition local to current MPI process
  virtual double compute_dt_local();

  //! Check if current time is larger than end time.
  virtual int finished();

  //! control how often AMR cycle is done (how many time steps between two AMR cycles)
  int m_amr_cycle_frequency;
  
  //! Check if AMR cycle is required
  virtual bool should_do_amr_cycle();

  /**
   * specify how often load balancing must be done.
   * Load balancing is performed once every amr_load_balancing_frequency
   * This value must be strictly larger than 0.
   */
  int m_amr_load_balancing_frequency;
  
  //! Check if Load Balancing is required
  virtual bool should_do_load_balancing();

  //! This is where action takes place. Wrapper arround next_iteration_impl.
  virtual void next_iteration();

  //! This is the next iteration computation (application specific).
  virtual void next_iteration_impl();

  //! This is were the time loop is done
  virtual void run();
  
  //! Decides if the current time step is eligible for dump data to file
  virtual bool should_save_solution();

  //! main routine to dump solution to file
  virtual void save_solution();

  //! main routine to dump solution to file
  virtual void save_solution_impl();

  //! should current time step write a restart file ?
  virtual bool should_write_restart_file();

  //! write restart file
  virtual void write_restart_file();
  
  //! read restart data
  virtual void read_restart_file();

  //! print monitoring information
  virtual void print_monitoring_info();

  /* IO related */

  //! counter incremented each time an output is written
  int m_times_saved;

  //! counter incremented each time a restart file is written
  int m_times_saved_restart;
  
  //! Number of variables to saved
  //int m_write_variables_number;

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
  
}; // class SolverBase

} // namespace dyablo

#endif // SOLVER_BASE_H_
