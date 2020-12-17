#pragma once

#include <map>
#include <string>
#include "shared/kokkos_shared.h"

#ifdef KOKKOS_ENABLE_CUDA
#include "CudaTimer.h"
#else
#include "OpenMPTimer.h"
#endif

class Timers{
public:
#ifdef KOKKOS_ENABLE_CUDA
  using Timer = CudaTimer;
#else
  using Timer = OpenMPTimer;
#endif

  Timers();
  ~Timers();

  /// Get Timer associated to name
  Timer& get(const std::string& name);
  /// Print a summary of all the timers
  void print();

private:
  std::map<std::string, Timer> timer_map;
};