#pragma once

#include <map>
#include <string>
#include <memory>

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
  std::unique_ptr<Timers_Timer_pimpl> data;
};

class Timers{
  public:
    using Timer = Timers_Timer;
    Timers();
    ~Timers();
    /// Get Timer associated to name
    Timer& get(const std::string& name);
    /// Print a summary of all the timers
    void print();
  private:
    std::unique_ptr<Timers_pimpl> data;
};