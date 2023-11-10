#include "Timers.h"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <fstream>

#include "kokkos_shared.h"
#include "utils/misc/Dyablo_assert.h"
#include "utils/mpi/GlobalMpiSession.h"

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
  Timers_Timer get(const std::string& name)
  {
    return Timers_Timer_pimpl();
  }
  /// Print a summary of all the timers
  void print()
  {
    std::cout << "(Timers are disabled)" << std::endl;
  }
  void print_file()
  {
    /*empty*/
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
  Timers_pimpl& parent_timers;

  OpenMPTimer omp_timer;
  #ifdef KOKKOS_ENABLE_CUDA
  std::unique_ptr<CudaTimer> cuda_timer;
  #endif
  std::map<std::string, Timers_Timer_pimpl> timer_map;
  Timers_Timer_pimpl* parent_timer = nullptr;

  /// Methods

  Timers_Timer_pimpl(const std::string& name, Timers_pimpl& parent_timers)
  : name(name), parent_timers(parent_timers), cuda_timer(std::make_unique<CudaTimer>())
  {}

  void start_timer_parent();

  void stop_timer_parent();

  void start()
  {
    start_timer_parent();

    Kokkos::Profiling::pushRegion(this->name);
    this->omp_timer.start();
    #ifdef KOKKOS_ENABLE_CUDA
    this->cuda_timer->start();
    #endif
  }

  void stop()
  {
    stop_timer_parent();

    Kokkos::Profiling::popRegion();
    this->omp_timer.stop();
    #ifdef KOKKOS_ENABLE_CUDA
    this->cuda_timer->stop();
    #endif
  }

  using Elapsed_mode_t = Timers_Timer::Elapsed_mode_t;
  
  double elapsed(Elapsed_mode_t em) const
  {
    DYABLO_ASSERT_HOST_RELEASE(parent_timer==nullptr, "Attempting to fetch a timer's elapsed time but the timer is still running : `" << name << "`");

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
  Timers_Timer_pimpl timer_total;
  Timers_Timer_pimpl* current_timer;


  // Methods
  Timers_pimpl()
  : timer_total("Total",*this)
  {
    timer_total.start();
  }

  ~Timers_pimpl()
  {
    timer_total.stop();
  }

  /// Get Timer associated to name
  Timers_Timer get(const std::string& name)
  {
    if( name == this->current_timer->name )
      return Timers_Timer{this->current_timer};

    std::map<std::string, Timers_Timer_pimpl>& timer_map = this->current_timer->timer_map;
    auto it = timer_map.find(name);
    if( it == timer_map.end() )
    {
      it = timer_map.emplace(std::piecewise_construct, 
                                  std::forward_as_tuple(name),
                                  std::forward_as_tuple(Timers_Timer_pimpl(name, *this)) ).first; 
    }
    return Timers_Timer{&it->second};
  }
  
  struct print_times_t
  {
    real_t time_cpu, time_gpu;
  };

  print_times_t print_aux( const Timers_Timer_pimpl& timer, std::string prefix )
  {
    using Mode = Timers_Timer::Elapsed_mode_t;
    double t_tot_CPU = timer_total.elapsed(Mode::ELAPSED_CPU);
    #ifdef KOKKOS_ENABLE_CUDA
    double t_tot_GPU = timer_total.elapsed(Mode::ELAPSED_GPU);
    #endif

    auto print_timer = [&]( std::string name, print_times_t times, std::string prefix )
    {
      std::cout << prefix << std::left << std::setw(25) << name;
      //int prefix_align_w = 10;
      //std::cout << std::setw(prefix_align_w-prefix.length()) << ""; // Align timers
      {
        double time_CPU = times.time_cpu;
        double percent_CPU = 100 * time_CPU / t_tot_CPU;
        printf(" time (CPU) : %9.3f s (%6.2f%%)", time_CPU, percent_CPU);
      }
      #ifdef KOKKOS_ENABLE_CUDA
      {
        double time_GPU = times.time_gpu;
        double percent_GPU = 100 * time_GPU / t_tot_GPU;
        printf(" , (GPU) : %9.3f s (%6.2f%%)", time_GPU, percent_GPU);
      }
      #endif 
      std::cout << std::endl;  
    };

    print_times_t times_current;
    times_current.time_cpu = timer.elapsed(Mode::ELAPSED_CPU);
    #ifdef KOKKOS_ENABLE_CUDA
    times_current.time_gpu = timer.elapsed(Mode::ELAPSED_GPU);
    #endif
    print_timer(timer.name, times_current, prefix);

    print_times_t sub_times_total{};
    std::string sub_prefix = prefix + "| ";
    //std::string sub_prefix = prefix + timer.name + "/";
    for( const auto& p : timer.timer_map )
    {
      print_times_t sub_times = print_aux(p.second, sub_prefix);
      sub_times_total.time_cpu += sub_times.time_cpu;
      sub_times_total.time_gpu += sub_times.time_gpu;
    }
    if( timer.timer_map.size() > 0 )
    {
      print_times_t times_other{ 
        times_current.time_cpu - sub_times_total.time_cpu,
        times_current.time_gpu - sub_times_total.time_gpu };
      print_timer("other", times_other, sub_prefix);
    }


    return times_current;
  }
  
  void get_timers_aux(const Timers_Timer_pimpl& timer, std::vector<std::string>& names, std::vector<real_t>& cpu_times_local)
  {
    using Mode = Timers_Timer::Elapsed_mode_t;
    names.push_back( timer.name );
    cpu_times_local.push_back( timer.elapsed(Mode::ELAPSED_CPU) );
    for( const auto& p : timer.timer_map )
    {
      get_timers_aux(p.second, names, cpu_times_local);
    }
  }

  void print_file()
  {
    dyablo::MpiComm mpi_comm = dyablo::GlobalMpiSession::get_comm_world();
    int tag = 10;

    std::vector<std::string> names;
    std::vector<real_t> cpu_times_local;

    timer_total.stop();
    get_timers_aux( timer_total, names, cpu_times_local );
    timer_total.start();

    if( mpi_comm.MPI_Comm_rank() != 0 )
      mpi_comm.MPI_Send(cpu_times_local.data(), cpu_times_local.size(), 0, tag);
    else
    {
      std::ofstream out("timers.txt");
      out << "Rank"; 
      for( const std::string& name : names )
      {
        out << " ; " << name;
      }
      out << std::endl;
      out << 0;
      for( real_t time : cpu_times_local )
      {
        out << " ; " << time;
      }
      out << std::endl;
      for( int r=1; r<mpi_comm.MPI_Comm_size(); r++ )
      {
        std::vector<real_t> cpu_times_remote(cpu_times_local.size());
        mpi_comm.MPI_Recv(cpu_times_remote.data(), cpu_times_remote.size(), r, tag);
        out << r;
        for( real_t time : cpu_times_remote )
        {
          out << " ; " << time;
        }
        out << std::endl;
      }
    }  
    
  }

  /// Print a summary of all the timers
  void print()
  {
    timer_total.stop();
    print_aux( timer_total, "" );
    timer_total.start();
  }
};
    
void Timers_Timer_pimpl::start_timer_parent()
{
  DYABLO_ASSERT_HOST_RELEASE(parent_timer == nullptr, "Attempting to start a timer that is already running : `" << name << "`");
  parent_timer = parent_timers.current_timer;
  parent_timers.current_timer = this;
}

void Timers_Timer_pimpl::stop_timer_parent()
{
  DYABLO_ASSERT_HOST_RELEASE(parent_timers.current_timer == this, "Attempting to stop a timer that is not running : `" << name << "`");
  parent_timers.current_timer = parent_timer;
  parent_timer = nullptr;
}

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

Timers_Timer Timers::get(const std::string& name)
{
  return data->get(name);
}

void Timers::print()
{
  data->print();
}

void Timers::print_file()
{
  data->print_file();
}
