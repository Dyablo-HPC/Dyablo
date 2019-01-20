#include "shared/SolverBase.h"

#include "shared/utils.h"

#ifdef USE_MPI
//#include "shared/mpiBorderUtils.h"
#endif // USE_MPI

//#include "utils/io/IO_ReadWrite.h"

namespace euler_pablo {

// =======================================================
// ==== CLASS SolverBase IMPL ============================
// =======================================================

// =======================================================
// =======================================================
SolverBase::SolverBase (HydroParams& params, ConfigMap& configMap) :
  params(params),
  configMap(configMap),
  solver_type(SOLVER_UNDEFINED)
{

  /*
   * init some variables by reading the ini parameter file.
   */
  read_config();

  /*
   * other variables initialization.
   */
  m_times_saved = 0;
  m_nCells = -1;
  m_nDofsPerCell = -1;
  
  // create the timers
  timers[TIMER_TOTAL]      = std::make_shared<Timer>();
  timers[TIMER_IO]         = std::make_shared<Timer>();
  timers[TIMER_DT]         = std::make_shared<Timer>();
  timers[TIMER_BOUNDARIES] = std::make_shared<Timer>();
  timers[TIMER_NUM_SCHEME] = std::make_shared<Timer>();

  // init variables names
  m_variables_names[ID] = "rho";
  m_variables_names[IP] = "energy";
  m_variables_names[IU] = "rho_vx"; // momentum component X
  m_variables_names[IV] = "rho_vy"; // momentum component Y
  m_variables_names[IW] = "rho_vz"; // momentum component Z
  m_variables_names[IA] = "bx"; // mag field X
  m_variables_names[IB] = "by"; // mag field Y
  m_variables_names[IC] = "bz"; // mag field Z
  
#ifdef USE_MPI
  const int gw = params.ghostWidth;
  const int isize = params.isize;
  const int jsize = params.jsize;
  const int ksize = params.ksize;
  const int nbvar = params.nbvar;

  // TODO
  
#endif // USE_MPI
  
} // SolverBase::SolverBase

// =======================================================
// =======================================================
SolverBase::~SolverBase()
{

  // m_io_reader_writer is now a shared (managed) pointer
  //delete m_io_reader_writer;
  
} // SolverBase::~SolverBase

// =======================================================
// =======================================================
void
SolverBase::read_config()
{

  m_t     = configMap.getFloat("run", "tCurrent", 0.0);
  m_tEnd  = configMap.getFloat("run", "tEnd", 0.0);
  m_dt    = m_tEnd;
  m_cfl   = configMap.getFloat("hydro", "cfl", 1.0);
  m_nlog  = configMap.getFloat("run", "nlog", 10);
  m_iteration = 0;

  m_problem_name = configMap.getString("hydro", "problem", "unknown");

  m_solver_name = configMap.getString("run", "solver_name", "unknown");

  /* restart run : default is no */
  m_restart_run_enabled = configMap.getInteger("run", "restart_enabled", 0);
  m_restart_run_filename = configMap.getString ("run", "restart_filename", "");

} // SolverBase::read_config

// =======================================================
// =======================================================
void
SolverBase::compute_dt()
{

#ifdef USE_MPI

  // get local time step
  double dt_local = compute_dt_local();

  // synchronize all MPI processes
  params.communicator->synchronize();

  // perform MPI_Reduceall to get global time step
  double dt_global;
  params.communicator->allReduce(&dt_local, &dt_global, 1, params.data_type, hydroSimu::MpiComm::MIN);

  m_dt = dt_global;
  
#else

  m_dt = compute_dt_local();
  
#endif

  // correct m_dt if necessary
  if (m_t+m_dt > m_tEnd) {
    m_dt = m_tEnd - m_t;
  }

} // SolverBase::compute_dt

// =======================================================
// =======================================================
double
SolverBase::compute_dt_local()
{

  // the actual numerical scheme must provide it a genuine implementation

  return m_tEnd;
  
} // SolverBase::compute_dt_local

// =======================================================
// =======================================================
int
SolverBase::finished()
{

  return m_t >= (m_tEnd - 1e-14) || m_iteration >= params.nStepmax;
  
} // SolverBase::finished

// =======================================================
// =======================================================
void
SolverBase::next_iteration()
{

  // setup a timer here (?)
  
  // genuine implementation called here
  next_iteration_impl();

  // perform some stats here (?)
  
  // incremenent
  ++m_iteration;
  m_t += m_dt;

} // SolverBase::next_iteration

// =======================================================
// =======================================================
void
SolverBase::next_iteration_impl()
{

  // This is application dependent
  
} // SolverBase::next_iteration_impl

// =======================================================
// =======================================================
void
SolverBase::save_solution()
{

  // save solution to output file
  save_solution_impl();
  
  // increment output file number
  ++m_times_saved;
  
} // SolverBase::save_solution

// =======================================================
// =======================================================
void
SolverBase::save_solution_impl()
{
} // SolverBase::save_solution_impl

// =======================================================
// =======================================================
void
SolverBase::read_restart_file()
{

  // TODO
  
} // SolverBase::read_restart_file

// =======================================================
// =======================================================
int
SolverBase::should_save_solution()
{
  
  double interval = m_tEnd / params.nOutput;

  // params.nOutput == 0 means no output at all
  // params.nOutput < 0  means always output 
  if (params.nOutput < 0) {
    return 1;
  }

  if ((m_t - (m_times_saved - 1) * interval) > interval) {
    return 1;
  }

  /* always write the last time step */
  if (ISFUZZYNULL (m_t - m_tEnd)) {
    return 1;
  }

  return 0;
  
} // SolverBase::should_save_solution

// =======================================================
// =======================================================
void
SolverBase::save_data(DataArray             U,
		      DataArray::HostMirror Uh,
		      int iStep,
		      real_t time)
{
  // TODO
}
// =======================================================
// =======================================================
void
SolverBase::save_data_debug(DataArray             U,
			    DataArray::HostMirror Uh,
			    int iStep,
			    real_t time,
			    std::string debug_name)
{
  // TODO
}

// =======================================================
// =======================================================
void
SolverBase::load_data(DataArray             U,
		      DataArray::HostMirror Uh,
		      int& iStep,
		      real_t& time)
{
  // TODO
}

// =======================================================
// =======================================================
void
SolverBase::make_boundary(DataArray Udata, FaceIdType faceId, bool mhd_enabled)
{

  // TODO
  
} // SolverBase::make_boundary


// =======================================================
// =======================================================
void
SolverBase::make_boundaries_serial(DataArray Udata, bool mhd_enabled)
{

  // TODO
  
} // SolverBase::make_boundaries_serial   

#ifdef USE_MPI
// =======================================================
// =======================================================
void
SolverBase::make_boundaries_mpi(DataArray Udata, bool mhd_enabled)
{

  // TODO
  
} // SolverBase::make_boundaries_mpi

#endif // USE_MPI

} // namespace euler_pablo
