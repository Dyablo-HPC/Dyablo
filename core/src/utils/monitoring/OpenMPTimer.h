/**
 * \file OpenMPTimer.h
 * \brief A simple timer class.
 *
 * \author Pierre Kestener
 * \date 29 Oct 2010
 *
 */
#ifndef DYABLO_UTILS_MONITORING_OPENMP_TIMER_H_
#define DYABLO_UTILS_MONITORING_OPENMP_TIMER_H_

#include <omp.h>

/**
 * \brief a simple Timer class.
 * If MPI is enabled, should we use MPI_WTime instead of gettimeofday (?!?)
 */
class OpenMPTimer
{
public:
  //! default constructor, timing starts rightaway
  OpenMPTimer();

  //! initialize total time
  OpenMPTimer(double t);

  //! copy constructor
  OpenMPTimer(const OpenMPTimer& aTimer);
  virtual ~OpenMPTimer();

  //! start time measure
  virtual void start();

  //! stop time measure and add result to total_time
  virtual void stop();

  //! return elapsed time in seconds (as stored in total_time)
  virtual double elapsed() const;

protected:
  double    start_time;

  //! store total accumulated timings
  double    total_time;

}; // class OpenMPTimer

#endif // DYABLO_UTILS_MONITORING_OPENMP_TIMER_H_
