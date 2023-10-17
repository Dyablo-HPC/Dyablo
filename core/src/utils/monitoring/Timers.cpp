#include "Timers.h"

#include <iostream>
#include <iomanip>
#include <cassert>

#include "kokkos_shared.h"
#include "utils/misc/Dyablo_assert.h"

#include "OpenMPTimer.h"
#ifdef KOKKOS_ENABLE_CUDA
#include "CudaTimer.h"
#endif

#ifdef DYABLO_DISABLE_TIMERS

/*-----------------------------
 * Timers Disabled
-------------------------------*/

struct Timers_Timer_pimpl
{
  void start(){}
  void stop(){}
  using Elapsed_mode_t = Timers_Timer::Elapsed_mode_t;
  double elapsed(Elapsed_mode_t mode) const{ return 0.0; }
};

struct Timers_pimpl
{
  /// Get Timer associated to name
  Timers_Timer& get(const std::string& name)
  {
    static Timers_Timer dummy = Timers_Timer_pimpl();
    return dummy;
  }
  /// Print a summary of all the timers
  void print()
  {
    std::cout << "(Timers are disabled)" << std::endl;
  }
};

#else //DYABLO_DISABLE_TIMERS
/*-----------------------------
 * Timers enabled 
-------------------------------*/

struct Timers_Timer_pimpl
{
  /// Members
  std::string name;
  OpenMPTimer omp_timer;
  #ifdef KOKKOS_ENABLE_CUDA
  std::unique_ptr<CudaTimer> cuda_timer;
  #endif
  bool running = false;

  /// Methods

  Timers_Timer_pimpl(const std::string& name)
  : name(name), cuda_timer(std::make_unique<CudaTimer>())
  {}

  void start()
  {
    DYABLO_ASSERT_HOST_RELEASE(!this->running, "Attempting to start a timer that is already running : `" << name << "`");
    this->running=true;

    Kokkos::Profiling::pushRegion(this->name);
    this->omp_timer.start();
    #ifdef KOKKOS_ENABLE_CUDA
    this->cuda_timer->start();
    #endif
  }

  void stop()
  {
    DYABLO_ASSERT_HOST_RELEASE(this->running, "Attempting to stop a timer that is not running : `" << name << "`");
    this->running=false;

    Kokkos::Profiling::popRegion();
    this->omp_timer.stop();
    #ifdef KOKKOS_ENABLE_CUDA
    this->cuda_timer->stop();
    #endif
  }

  using Elapsed_mode_t = Timers_Timer::Elapsed_mode_t;
  
  double elapsed(Elapsed_mode_t em) const
  {
    DYABLO_ASSERT_HOST_RELEASE(!this->running, "Attempting to fetch a timer's elapsed time but the timer is still running : `" << name << "`");

    if(em == Timers_Timer::ELAPSED_CPU)
      return this->omp_timer.elapsed();
    #ifdef KOKKOS_ENABLE_CUDA
    else if( em == Timers_Timer::ELAPSED_GPU )
      return this->cuda_timer->elapsed();
    #endif
    else
    {
      DYABLO_ASSERT_HOST_RELEASE(false, "Unknown/incompatible elapsed mode" );
      return 0;
    }
  }
};

struct Timers_pimpl
{
  /// Members
  std::map<std::string, Timers_Timer> timer_map;
  Timers_Timer timer_total;


  // Methods
  Timers_pimpl()
  : timer_total{std::make_unique<Timers_Timer_pimpl>("Total")}
  {
    timer_total.start();
  }

  ~Timers_pimpl()
  {
    timer_total.stop();
  }

  /// Get Timer associated to name
  Timers_Timer& get(const std::string& name)
  {
    auto it = this->timer_map.find(name);
    if( it == this->timer_map.end() )
    {
      Timers_Timer timer{ std::make_unique<Timers_Timer_pimpl>(name) };
      it = this->timer_map.emplace(std::piecewise_construct, 
                                  std::forward_as_tuple(name),
                                  std::forward_as_tuple(std::move(timer)) ).first; 
    }
    return it->second;
  }
  /// Print a summary of all the timers
  void print()
  {
    using Mode = Timers_Timer::Elapsed_mode_t;
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
};

#endif


/*-----------------------------
 * Delegate to pimpl
-------------------------------*/
void Timers_Timer::start()
{
  data->start();
}

void Timers_Timer::stop()
{
  data->stop();
}

double Timers_Timer::elapsed(Elapsed_mode_t mode) const
{
  return data->elapsed(mode);
}

Timers::Timers()
: data( std::make_unique<Timers_pimpl>() )
{}

Timers::~Timers(){}

Timers_Timer& Timers::get(const std::string& name)
{
  return data->get(name);
}

void Timers::print()
{
  data->print();
}
