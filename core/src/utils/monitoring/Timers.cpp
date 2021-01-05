#include "Timers.h"

#include <iostream>
#include <iomanip>
#include <cassert>

#include "shared/kokkos_shared.h"

#include "OpenMPTimer.h"
#ifdef KOKKOS_ENABLE_CUDA
#include "CudaTimer.h"
#endif

Timers::Timers()
: timer_total("Total")
{
  timer_total.start();
}
Timers::~Timers()
{
  timer_total.stop();
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
  using Mode = Timer::Elapsed_mode_t;
  timer_total.stop();
  double t_tot_CPU = timer_total.elapsed(Mode::ELAPSED_CPU);
  std::cout << "Total elapsed time (CPU) : " << t_tot_CPU;
  #ifdef KOKKOS_ENABLE_CUDA
  double t_tot_GPU = timer_total.elapsed(Mode::ELAPSED_GPU);
  std::cout << " , (GPU) : " << t_tot_GPU;
  #endif
  std::cout << std::endl;
  timer_total.start();
  
  for( const auto& p : this->timer_map )
  {
    const std::string& name = p.first;
    {
      double time_CPU = p.second.elapsed(Mode::ELAPSED_CPU);
      double percent_CPU = 100 * time_CPU / t_tot_CPU;
      std::cout << std::left << std::setw(25) << name;
      printf(" time (CPU) : %5.3f \ts (%5.2f%%)", time_CPU, percent_CPU);
    }
    #ifdef KOKKOS_ENABLE_CUDA
    {
      double time_GPU = p.second.elapsed(Mode::ELAPSED_GPU);
      double percent_GPU = 100 * time_GPU / t_tot_GPU;
      printf(" , (GPU) : %5.3f \ts (%5.2f%%)", time_GPU, percent_GPU);
    }
    #endif 
    std::cout << std::endl;  
  }
}

struct Timers::Timer::Timer_pimpl{
  OpenMPTimer omp_timer;
  #ifdef KOKKOS_ENABLE_CUDA
  CudaTimer cuda_timer;
  #endif
  bool running = false;
};

Timers::Timer::Timer(const std::string& name)
: name(name), data(std::make_unique<Timer_pimpl>())
{}

void Timers::Timer::start()
{
  assert(!data->running);
  data->running=true;

  Kokkos::Profiling::pushRegion(this->name);
  data->omp_timer.start();
  #ifdef KOKKOS_ENABLE_CUDA
  data->cuda_timer.start();
  #endif
}

void Timers::Timer::stop()
{
  assert(data->running);
  data->running=false;

  Kokkos::Profiling::popRegion();
  data->omp_timer.stop();
  #ifdef KOKKOS_ENABLE_CUDA
  data->cuda_timer.stop();
  #endif
}

double Timers::Timer::elapsed(Timers::Timer::Elapsed_mode_t em) const
{
  assert(!data->running);

  if(em == ELAPSED_CPU)
    return data->omp_timer.elapsed();
  #ifdef KOKKOS_ENABLE_CUDA
  else if( em == ELAPSED_GPU )
    return data->cuda_timer.elapsed();
  #endif
  else
  {
    assert(false); // Unknown/incompatible elapsed mode
    return 0;
  }
}