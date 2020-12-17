#include "Timers.h"

#include <iostream>
#include <iomanip>

#include "shared/kokkos_shared.h"

#include "OpenMPTimer.h"
#ifdef KOKKOS_ENABLE_CUDA
#include "CudaTimer.h"
#endif

Timers::Timers()
{
  get("___TOTAL___").start();
}
Timers::~Timers()
{
  get("___TOTAL___").stop();
}

Timers::Timer& Timers::get(const std::string& name)
{
  auto it = this->timer_map.find(name);
  if( it == this->timer_map.end() )
    it = this->timer_map.emplace(std::piecewise_construct, 
                                std::forward_as_tuple(name),
                                std::forward_as_tuple(name) ).first;
  
  return it->second;
}

void Timers::print()
{
  Timers::Timer& timer_tot = get("___TOTAL___");
  timer_tot.stop();
  double t_tot = timer_tot.elapsed();
  timer_tot.start();
  std::cout << "Total elapsed time : " << t_tot << std::endl;

  for( const auto& p : this->timer_map )
  {
    const std::string& name = p.first;
    if( name != "___TOTAL___" )
    {
      double time = p.second.elapsed();
      double percent = 100 * time / t_tot;
      std::cout << std::left << std::setw(25) << name;
      printf(" time : %5.3f secondes %5.2f%%\n", time, percent);      
    }
  }
}

struct Timers::Timer::Timer_pimpl{
  OpenMPTimer omp_timer;
  #ifdef KOKKOS_ENABLE_CUDA
  CudaTimer cuda_timer;
  #endif
};

Timers::Timer::Timer(const std::string& name)
: name(name), data(std::make_unique<Timer_pimpl>())
{}

void Timers::Timer::start()
{
  Kokkos::Profiling::pushRegion(this->name);
  data->omp_timer.start();
  #ifdef KOKKOS_ENABLE_CUDA
  data->cuda_timer.start();
  #endif
}

void Timers::Timer::stop()
{
  Kokkos::Profiling::popRegion();
  data->omp_timer.stop();
  #ifdef KOKKOS_ENABLE_CUDA
  data->cuda_timer.stop();
  #endif
}

double Timers::Timer::elapsed() const
{
  #ifdef KOKKOS_ENABLE_CUDA
  return data->cuda_timer.elapsed();
  #else
  return data->omp_timer.elapsed();
  #endif
}