#pragma once 

#include "utils/misc/RegisteringFactory.h"

namespace dyablo{
namespace muscl_block{

class SolverHydroMusclBlock;

class InitialConditions{
public:
  //InitialConditions();
  virtual void init(SolverHydroMusclBlock* solver) = 0;
  virtual ~InitialConditions(){}
};

using InitialConditionsFactory = RegisteringFactory<InitialConditions>;

} // namespace muscl_block
} // namespace dyablo