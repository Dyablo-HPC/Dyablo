/**
 * \file PapiInfo.cpp
 * \brief A simple PAPI interface class.
 *
 * Parts of this class is inspired by file sc_flops.c found in library
 * libsc (https://github.com/cburstedde/libsc).
 *
 * \author Pierre Kestener
 * \date March 3rd, 2014
 *
 */
#include "PapiInfo.h"

#include <time.h>
#include <sys/time.h> // for gettimeofday and struct timeval

#include <stdio.h>
#include <papi.h>

namespace ppkMHD {

////////////////////////////////////////////////////////////////////////////////
// PapiInfo class methods body
////////////////////////////////////////////////////////////////////////////////

// =======================================================
// =======================================================
PapiInfo::PapiInfo() : papiTimer() {
  
  crtime  = 0.0f;
  cptime  = 0.0f;
  cflpops = 0;
  irtime  = 0.0f;
  iptime  = 0.0f;
  iflpops = 0;
  mflops  = 0.0;
  float tmp;
  
  // initialize PAPI counters
  PAPI_flops (&irtime, &iptime, &iflpops, &tmp);
  
} // PapiInfo::PapiInfo

// =======================================================
// =======================================================
PapiInfo::~PapiInfo()
{
  
} // PapiInfo::~PapiInfo

// =======================================================
// =======================================================
void PapiInfo::start()
{
  
  float tmp;
  int retval;
  
  papiTimer.start();
  if ( (retval=PAPI_flops (&irtime, &iptime, &iflpops, &tmp)) < PAPI_OK)
    printf("PAPI not ok in PapiInfoStart with returned value %d\n",retval);
  
} // PapiInfo::start

// =======================================================
// =======================================================
void PapiInfo::stop() {
  
  float rtime, ptime;
  long long int flpops;
  float tmp;
  int retval;
  
  if ( (retval=PAPI_flops (&rtime, &ptime, &flpops, &tmp)) < PAPI_OK)
    printf("PAPI not ok in PapiInfoStop with returned value %d\n",retval);
  papiTimer.stop();
  
  // add increment from previous call to start values to accumulator counters
  crtime  = rtime  - irtime;
  cptime  = ptime  - iptime;
  cflpops += flpops - iflpops;
  
  mflops = 1.0 * cflpops / papiTimer.elapsed();
  
} // PapiInfo::stop

// =======================================================
// =======================================================
double PapiInfo::getFlops() {
  
  return mflops;
  
} // PapiInfo::getFlops

// =======================================================
// =======================================================
long long int PapiInfo::getFlop() {
  
  return cflpops;
  
} // PapiInfo::getFlop

// =======================================================
// =======================================================
double PapiInfo::elapsed() {
  
  return papiTimer.elapsed();
  
} // PapiInfo::elapsed

} // namespace ppkMHD
