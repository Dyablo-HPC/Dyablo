#include "utils/misc/RegisteringFactory.h"

struct Base{
  virtual double test() = 0;
};

template <typename T>
struct Impl2 : public Base
{
  T val;
  Impl2(const T& val) : val(val) {}
  double test(){
    return val+0.5;
  }
};

using BaseFactory = RegisteringFactory<Base, double>;

FACTORY_REGISTER(BaseFactory, Impl2<double>, "double2");

