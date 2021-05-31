#include <boost/test/unit_test.hpp>

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


BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_CASE(test_RegisteringFactory)
{
  std::unique_ptr<Base> p_int = BaseFactory::make_instance("int", 5);
  std::unique_ptr<Base> p_double = BaseFactory::make_instance("double", 3.5);
  std::unique_ptr<Base> p_double2 = BaseFactory::make_instance("double2", 4);

  BOOST_CHECK_CLOSE(5, p_int->test(), 0.001);;
  BOOST_CHECK_CLOSE(3.5, p_double->test(), 0.001);;
  BOOST_CHECK_CLOSE(4.5, p_double2->test(), 0.001);;
}

BOOST_AUTO_TEST_SUITE_END() /* dyablo */
