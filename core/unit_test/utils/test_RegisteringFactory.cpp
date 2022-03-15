#include "gtest/gtest.h"

#include "utils/misc/RegisteringFactory.h"

struct Base{
  virtual double test() = 0;
};

template <typename T>
struct Impl : public Base
{
  T val;
  Impl(const T& val) : val(val) {}
  double test(){
    return val;
  }
};

using BaseFactory = RegisteringFactory<Base, double>;

FACTORY_REGISTER(BaseFactory, Impl<int>, "int");
FACTORY_REGISTER(BaseFactory, Impl<double>, "double");

// Impl2 is registered in separate RegisteringFactory_impl_to_register.cpp
template <typename T> struct Impl2;

template<>
bool BaseFactory::init()
{
  DECLARE_REGISTERED(Impl<int>);
  DECLARE_REGISTERED(Impl<double>);
  DECLARE_REGISTERED(Impl2<double>);

  return true;
}

TEST(dyablo, test_RegisteringFactory)
{
  std::unique_ptr<Base> p_int = BaseFactory::make_instance("int", 5);
  std::unique_ptr<Base> p_double = BaseFactory::make_instance("double", 3.5);
  std::unique_ptr<Base> p_double2 = BaseFactory::make_instance("double2", 4);

  EXPECT_EQ( 5, p_int->test() );
  EXPECT_DOUBLE_EQ( 3.5, p_double->test() );
  EXPECT_DOUBLE_EQ( 4.5, p_double2->test() );
}
