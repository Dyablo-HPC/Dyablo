#include "gtest/gtest.h"

#include "utils/misc/RegisteringFactory.h"

struct Base{
  virtual double test() = 0;
  virtual ~Base(){}
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

TEST( Test_RegisteringFactory, available_ids_count )
{
  auto ids = BaseFactory::get_available_ids();
  EXPECT_EQ( 3, ids.size() );

  std::cout << "ids: " << std::endl;
  for(size_t i=0; i<ids.size(); i++)
    std::cout << "\'" << ids[i] << "\'" << std::endl;
}

TEST( Test_RegisteringFactory, same_compile_unit )
{
  std::unique_ptr<Base> p_int = BaseFactory::make_instance("int", 5);
  std::unique_ptr<Base> p_double = BaseFactory::make_instance("double", 3.5);

  EXPECT_DOUBLE_EQ( 5, p_int->test() );
  EXPECT_DOUBLE_EQ( 3.5, p_double->test() );
}

TEST( Test_RegisteringFactory, different_compile_unit )
{
  std::unique_ptr<Base> p_double2 = BaseFactory::make_instance("double2", 4);

  EXPECT_DOUBLE_EQ( 4.5, p_double2->test() );
}
