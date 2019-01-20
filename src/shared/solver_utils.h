#ifndef SOLVER_UTILS_H_
#define SOLVER_UTILS_H_

#include "shared/SolverBase.h"

namespace euler_pablo {

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

  int myRank = 0;
  int nProcs = 1;

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
  
    printf("Perf             : %5.3f number of Mcell-updates/s\n",solver->m_iteration*solver->m_nCells*nProcs/t_tot*1e-6);

  } // end myRank==0

} // print_solver_monitoring_info

} // namespace euler_pablo

#endif // SOLVER_UTILS_H_