#pragma once

#include <map>
#include <string>
#include <memory>

class Timers{
public:
  class Timer{
  public:
    Timer(const std::string& name);
    void start();
    void stop();
    enum Elapsed_mode_t
    {
      ELAPSED_CPU,
      ELAPSED_GPU
    };
    double elapsed(Elapsed_mode_t mode) const;
  private:
    struct Timer_pimpl;
    std::string name;
    std::unique_ptr<Timer_pimpl> data;
  };

  Timers();
  ~Timers();

  /// Get Timer associated to name
  Timer& get(const std::string& name);
  /// Print a summary of all the timers
  void print();

private:
  std::map<std::string, Timer> timer_map;
};