#include "Timers.h"

#include <iostream>
#include <iomanip>

Timers::Timers()
{
  timer_map["___TOTAL___"].start();
}
Timers::~Timers()
{
  timer_map.at("___TOTAL___").stop();
}

Timers::Timer& Timers::get(const std::string& name)
{
  return this->timer_map[name];
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