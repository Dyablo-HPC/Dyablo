#ifndef SOLVER_UTILS_H_
#define SOLVER_UTILS_H_

#include "shared/SolverBase.h"

namespace dyablo {

/**
 * print monitoring information
 */
inline void print_solver_monitoring_info(SolverBase* solver)
{
  
  real_t t_tot   = solver->m_timers[TIMER_TOTAL]->elapsed();
  real_t t_comp  = solver->m_timers[TIMER_NUM_SCHEME]->elapsed();
  real_t t_dt    = solver->m_timers[TIMER_DT]->elapsed();
  real_t t_bound = solver->m_timers[TIMER_BOUNDARIES]->elapsed();
  real_t t_io    = solver->m_timers[TIMER_IO]->elapsed();
  real_t t_amr   = solver->m_timers[TIMER_AMR_CYCLE]->elapsed();

  real_t t_amr_sync_ghost   = solver->m_timers[TIMER_AMR_CYCLE_SYNC_GHOST]->elapsed();
  real_t t_amr_mark_cells   = solver->m_timers[TIMER_AMR_CYCLE_MARK_CELLS]->elapsed();
  real_t t_amr_adapt_mesh   = solver->m_timers[TIMER_AMR_CYCLE_ADAPT_MESH]->elapsed();
  real_t t_amr_map_userdata = solver->m_timers[TIMER_AMR_CYCLE_MAP_USERDATA]->elapsed();
  real_t t_amr_load_balance = solver->m_timers[TIMER_AMR_CYCLE_LOAD_BALANCE]->elapsed();

  real_t t_block_copy = solver->m_timers[TIMER_BLOCK_COPY]->elapsed();

  int myRank = 0;
  int nProcs = 1;
  UNUSED(nProcs);

#ifdef USE_MPI
  myRank = solver->params.myRank;
  nProcs = solver->params.nProcs;
#endif // USE_MPI
  
  // only print on master
  if (myRank == 0) {

    printf("total       time : %5.3f secondes\n",t_tot);
    printf("godunov     time : %5.3f secondes %5.2f%%\n",t_comp,100*t_comp/t_tot);
    printf("compute dt  time : %5.3f secondes %5.2f%%\n",t_dt,100*t_dt/t_tot);
    printf("boundaries  time : %5.3f secondes %5.2f%%\n",t_bound,100*t_bound/t_tot);
    printf("io          time : %5.3f secondes %5.2f%%\n",t_io,100*t_io/t_tot);

    printf("block copy  time : %5.3f secondes %5.2f%%\n",t_block_copy,100*t_block_copy/t_tot);

    printf("amr cycle   time : %5.3f secondes %5.2f%%\n",t_amr,100*t_amr/t_tot);

    printf("amr cycle sync ghost    : %5.3f secondes %5.2f%%\n",t_amr_sync_ghost,100*t_amr_sync_ghost/t_tot);
    printf("amr cycle mark cells    : %5.3f secondes %5.2f%%\n",t_amr_mark_cells,100*t_amr_mark_cells/t_tot);
    printf("amr cycle adapt mesh    : %5.3f secondes %5.2f%%\n",t_amr_adapt_mesh,100*t_amr_adapt_mesh/t_tot);
    printf("amr cycle map user data : %5.3f secondes %5.2f%%\n",t_amr_map_userdata,100*t_amr_map_userdata/t_tot);
    printf("amr cycle load balance  : %5.3f secondes %5.2f%%\n",t_amr_load_balance,100*t_amr_load_balance/t_tot);

    printf("Perf             : %5.3f number of Mcell-updates/s\n",solver->m_total_num_cell_updates/t_tot*1e-6);
    
    printf("Total number of cell-updates : %ld\n",solver->m_total_num_cell_updates);

  } // end myRank==0

} // print_solver_monitoring_info

} // namespace dyablo

#endif // SOLVER_UTILS_H_
