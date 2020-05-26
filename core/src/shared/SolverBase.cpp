#include "shared/SolverBase.h"

#include "shared/utils.h"

#include <memory>

#ifdef USE_MPI
//#include "shared/mpiBorderUtils.h"
#endif // USE_MPI

namespace dyablo {

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

  // 2D or 3D ?
  m_dim = params.dimType == TWO_D ? 2 : 3;

  // create PABLO mesh
  amr_mesh = std::make_shared<AMRmesh>(m_dim);

  // set default behavior regarding 2:1 balance
  // codim 1 ==> balance through faces
  // codim 2 ==> balance through faces and corner
  // codim 3 ==> balance through faces, edges and corner (3D only)
  int codim = configMap.getInteger("amr", "codim", m_dim);
  amr_mesh->setBalanceCodimension(codim);

  uint32_t idx = 0;
  amr_mesh->setBalance(idx,true);


  // here periodic means : 
  // every cell will have at least one neighbor through every face
  // periodicity for user data is treated elsewhere, here we only
  // deal with periodicity at mesh level
  if (params.boundary_type_xmin == BC_PERIODIC)
    amr_mesh->setPeriodic(0);
  if (params.boundary_type_xmax == BC_PERIODIC)
    amr_mesh->setPeriodic(1);
  if (params.boundary_type_ymin == BC_PERIODIC)
    amr_mesh->setPeriodic(2);
  if (params.boundary_type_ymax == BC_PERIODIC)
    amr_mesh->setPeriodic(3);

  if (m_dim == 3) {
    if (params.boundary_type_zmin == BC_PERIODIC)
      amr_mesh->setPeriodic(4);
    if (params.boundary_type_zmax == BC_PERIODIC)
      amr_mesh->setPeriodic(5);
  }

  // set the number of children upon refinement
  m_nbChildren = m_dim == 2 ? 4 : 8;

  /*
   * Load balancing control
   */
  m_amr_load_balancing_frequency = configMap.getInteger("amr", "load_balancing_frequency", 1);
  if (m_amr_load_balancing_frequency < 1) {
    m_amr_load_balancing_frequency = 1;
  }

  /*
   * AMR cycle control
   */
  m_amr_cycle_frequency = configMap.getInteger("amr", "cycle_frequency", 1);
  if (m_amr_cycle_frequency < 1) {
    m_amr_cycle_frequency = 1;
  }

  // TODO : analyze if amr_cycle_frequency and 
  // amr_load_balancing_frequency can be completely independent, or
  // should we restrict / enforce e.g. load balacing frequency to be
  // multiple of amr cycle frequency ?

  /*
   * other variables initialization.
   */  
  m_times_saved = 0;
  m_times_saved_restart = 0;
  
  m_nCells = -1;
  m_nDofsPerCell = -1;

  // statistics
  m_total_num_cell_updates = 0;
  
  // create the timers - ugly - refactor me - TODO
  m_timers[TIMER_TOTAL]      = std::make_shared<Timer>();
  m_timers[TIMER_IO]         = std::make_shared<Timer>();
  m_timers[TIMER_DT]         = std::make_shared<Timer>();
  m_timers[TIMER_BOUNDARIES] = std::make_shared<Timer>();
  m_timers[TIMER_NUM_SCHEME] = std::make_shared<Timer>();
  m_timers[TIMER_AMR_CYCLE]  = std::make_shared<Timer>();
  m_timers[TIMER_AMR_CYCLE_SYNC_GHOST]  = std::make_shared<Timer>();
  m_timers[TIMER_AMR_CYCLE_MARK_CELLS]  = std::make_shared<Timer>();
  m_timers[TIMER_AMR_CYCLE_ADAPT_MESH]  = std::make_shared<Timer>();
  m_timers[TIMER_AMR_CYCLE_MAP_USERDATA]  = std::make_shared<Timer>();
  m_timers[TIMER_AMR_CYCLE_LOAD_BALANCE]  = std::make_shared<Timer>();
  m_timers[TIMER_AMR_BLOCK_COPY] = std::make_shared<Timer>();
  m_timers[TIMER_BLOCK_COPY] = std::make_shared<Timer>();

#ifdef USE_MPI
  //const int nbvar = params.nbvar;

  // TODO
  
#endif // USE_MPI
  
} // SolverBase::SolverBase

// =======================================================
// =======================================================
SolverBase::~SolverBase()
{

} // SolverBase::~SolverBase

// =======================================================
// =======================================================
void
SolverBase::do_amr_cycle()
{

  // Example of what must be implemented in derived class

  // 1. User data comm to update ghost cell values

  // 2. mark cell for refinement / coarsening

  // 3. adapt mesh

  // 4. user data repping
  
} // SolverBase::do_amr_cycle

// =======================================================
// =======================================================
void
SolverBase::do_load_balancing()
{

  // perform MPI load balancing (mesh + user data)

} // SolverBase::do_load_balancing

// =======================================================
// =======================================================
void
SolverBase::read_config()
{

  m_t     = configMap.getFloat("run", "tCurrent", 0.0);
  m_tBeg  = configMap.getFloat("run", "tCurrent", 0.0);
  m_tEnd  = configMap.getFloat("run", "tEnd", 0.0);
  m_max_iterations = params.nStepmax;

  // maximun number of output written
  m_max_save_count = params.nOutput;

  // maximun number of restart output written
  m_max_restart_count = configMap.getInteger("run", "restart_count", 1);

  
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

  // TODO : refactor me, please
  
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

  return m_t >= (m_tEnd - 1e-14) || m_iteration >= m_max_iterations;
  
} // SolverBase::finished

// =======================================================
// =======================================================
// TODO: better strategy to decide when to adapt ?
bool
SolverBase::should_do_amr_cycle()
{

  // default behavior : once every amr_cycle_frequency time steps 
  return params.amr_cycle_enabled and ( (m_iteration % m_amr_cycle_frequency) == 0);

} // SolverBase::should_do_amr_cycle

// =======================================================
// =======================================================
bool
SolverBase::should_do_load_balancing()
{

  // default behavior : true once every amr_load_balancing_frequency
  return (m_iteration % m_amr_load_balancing_frequency) == 0;

} // SolverBase::should_do_load_balancing

// =======================================================
// =======================================================
void
SolverBase::next_iteration()
{

  // setup a timer here (?)
  
  // genuine implementation called here
  next_iteration_impl();

  // perform some stats here (?)
  m_total_num_cell_updates += amr_mesh->getGlobalNumOctants();
  
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
SolverBase::run()
{

  /*
   * Default implementation for the time loop
   */
  while ( !finished() ) {

    next_iteration();
    
  } // end time loop
  
} // SolverBase::run

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
bool
SolverBase::should_write_restart_file()
{

  if (m_max_restart_count > 0) {

    double interval = m_tEnd / m_max_restart_count;

    // never write restart file at t = m_tBeg
    if (((m_t - m_tBeg) - m_times_saved_restart * interval) > interval) {
      return true;
    }
    
    /* always write the restart at the last time step */
    if (ISFUZZYNULL (m_t - m_tEnd)) {
      return true;
    }
    
  }

  return false;
  
} // SolverBase::should_write_restart

// =======================================================
// =======================================================
void
SolverBase::write_restart_file()
{

  // TODO
  if (params.myRank==0)
    std::cout << "Please implement me : SolverBase::write_restart_file\n";
  
} // SolverBase::write_restart_file

// =======================================================
// =======================================================
void
SolverBase::read_restart_file()
{

  // TODO
  
} // SolverBase::read_restart_file

// =======================================================
// =======================================================
void
SolverBase::print_monitoring_info()
{

  real_t t_tot   = m_timers[TIMER_TOTAL]->elapsed();
  real_t t_comp  = m_timers[TIMER_NUM_SCHEME]->elapsed();
  real_t t_dt    = m_timers[TIMER_DT]->elapsed();
  real_t t_bound = m_timers[TIMER_BOUNDARIES]->elapsed();
  real_t t_io    = m_timers[TIMER_IO]->elapsed();
  real_t t_amr   = m_timers[TIMER_AMR_CYCLE]->elapsed();

  real_t t_amr_sync_ghost   = m_timers[TIMER_AMR_CYCLE_SYNC_GHOST]->elapsed();
  real_t t_amr_mark_cells   = m_timers[TIMER_AMR_CYCLE_MARK_CELLS]->elapsed();
  real_t t_amr_adapt_mesh   = m_timers[TIMER_AMR_CYCLE_ADAPT_MESH]->elapsed();
  real_t t_amr_map_userdata = m_timers[TIMER_AMR_CYCLE_MAP_USERDATA]->elapsed();
  real_t t_amr_load_balance = m_timers[TIMER_AMR_CYCLE_LOAD_BALANCE]->elapsed();

  //real_t t_block_copy = m_timers[TIMER_BLOCK_COPY]->elapsed();

  int myRank = 0;
  int nProcs = 1;
  UNUSED(nProcs);

#ifdef USE_MPI
  myRank = params.myRank;
  nProcs = params.nProcs;
#endif // USE_MPI
  
  // only print on master
  if (myRank == 0) {

    printf("total       time : %5.3f secondes\n", t_tot);
    printf("godunov     time : %5.3f secondes %5.2f%%\n", t_comp,
           100 * t_comp / t_tot);
    printf("compute dt  time : %5.3f secondes %5.2f%%\n", t_dt,
           100 * t_dt / t_tot);
    printf("boundaries  time : %5.3f secondes %5.2f%%\n", t_bound,
           100 * t_bound / t_tot);
    printf("io          time : %5.3f secondes %5.2f%%\n", t_io,
           100 * t_io / t_tot);

    // printf("block copy  time : %5.3f secondes %5.2f%%\n", t_block_copy,
    //        100 * t_block_copy / t_tot);

    printf("amr cycle   time : %5.3f secondes %5.2f%%\n", t_amr,
           100 * t_amr / t_tot);

    printf("amr cycle sync ghost    : %5.3f secondes %5.2f%%\n",
           t_amr_sync_ghost, 100 * t_amr_sync_ghost / t_tot);
    printf("amr cycle mark cells    : %5.3f secondes %5.2f%%\n",
           t_amr_mark_cells, 100 * t_amr_mark_cells / t_tot);
    printf("amr cycle adapt mesh    : %5.3f secondes %5.2f%%\n",
           t_amr_adapt_mesh, 100 * t_amr_adapt_mesh / t_tot);
    printf("amr cycle map user data : %5.3f secondes %5.2f%%\n",
           t_amr_map_userdata, 100 * t_amr_map_userdata / t_tot);
    printf("amr cycle load balance  : %5.3f secondes %5.2f%%\n",
           t_amr_load_balance, 100 * t_amr_load_balance / t_tot);

    printf("Perf             : %5.3f number of Mcell-updates/s\n",
           m_total_num_cell_updates / t_tot * 1e-6);

    printf("Total number of cell-updates : %ld\n",
           m_total_num_cell_updates);

  } // end myRank==0

} // SolverBase::print_monitoring_info

// =======================================================
// =======================================================
bool
SolverBase::should_save_solution()
{
  
  double interval = (m_tEnd - m_tBeg) / params.nOutput;

  // params.nOutput == 0 means no output at all
  // params.nOutput < 0  means always output 
  if (m_max_save_count < 0) {
    return true;
  }

  if ((m_t - (m_times_saved - 1) * interval) > interval) {
    return true;
  }

  /* always write the last time step */
  if (ISFUZZYNULL (m_t - m_tEnd)) {
    return true;
  }

  return false;
  
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
  
} // SolverBase::load_data

} // namespace dyablo
