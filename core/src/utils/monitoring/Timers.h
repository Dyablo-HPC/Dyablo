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
    double elapsed() const;
  private:
    struct Timer_pimpl;
    std::unique_ptr<Timer_pimpl> data;
    std::string name;
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