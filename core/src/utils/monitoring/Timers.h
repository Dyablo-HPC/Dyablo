#pragma once

#include <map>
#include <string>
#include <memory>
#include <vector>

struct Timers_Timer_pimpl;
struct Timers_pimpl;

struct Timers_Timer{
  void start();
  void stop();
  enum Elapsed_mode_t
  {
    ELAPSED_CPU,
    ELAPSED_GPU
  };
  double elapsed(Elapsed_mode_t mode) const;
  Timers_Timer_pimpl* data;
};

class Timers{
  public:
    using Timer = Timers_Timer;
    Timers();
    ~Timers();
    /// Get Timer associated to name
    Timer get(const std::string& name);
    /// Print a summary of all the timers
    void print();
    void get_timers(std::vector<std::string>& names, std::vector<double>& cpu_times);
  private:
    std::unique_ptr<Timers_pimpl> data;
};