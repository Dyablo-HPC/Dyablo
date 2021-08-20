#pragma once

#include <memory>
#include <map>
#include <string>
#include <utility>
#include <stdexcept>
#include <tuple>
#include <iostream>

/**
 * A general factory to construct instances of BaseType from an id and constructor parameters
 * @tparam BaseType : Constructed instances are implementations of BaseType
 * @tparam Args... : implementations of basetype have a constructor compatible with Args...
 * 
 * When implementing new BaseType implementations, dynamic types are registered using the FACTORY_REGISTER() macro and linked to an identifier
 * During execution, specific implementations of BaseType are constructed using RegisteringFactory::make_instance(<identifier>, <constructor_params...>)
 **/
template< typename BaseType, typename... Args>
class RegisteringFactory{
public:
  /**
   * Must be implemented and contains the list of registered class (in a DECLARE_REGISTERED macro)
   * This will be called during the first call to make_instance() to verify the classes have actually been registered
   * 
   * e.g:
   * ```
   * class Impl1;
   * class Impl2;
   * class Impl3;
   * double ImplFactory::init()
   * {
   *    DECLARE_REGISTERED(Impl1);
   *    DECLARE_REGISTERED(Impl2);
   *    DECLARE_REGISTERED(Impl3);
   * 
   *    return true;
   * }
   * ```
   **/
  static bool init();

  /**
   * Construct an instance of BaseType with the actual dynamic type corresponding to the id registered with FACTORY_REGISTER
   * 
   * @param id a valid identifier from a FACTORY_REGISTER declaration, also listed in RegisteringFactory::init().
   * @param args... parameters for the constructor of the class to instanciate
   **/
  static std::unique_ptr<BaseType> make_instance(const std::string& id, Args... args )
  {
    init();

    auto& constructs = get_constructs(); 

    if( constructs.find(id) == constructs.end() )
    {
      std::cout << "Factory : \'"<<id<<"\' not found" << std::endl;
      std::cout << "Registered ids : (" << constructs.size() << ")" << std::endl;
      for(auto& p : constructs)
        std::cout << "  - \'" << p.first << "\'" << std::endl;

      throw std::runtime_error("Factory : \'"+id+"\' not found");
    }

    std::unique_ptr<BaseType> res = 
      constructs.at(id)->make_instance( args... );
    return res;
  }

  /// Register class (do not use directly, use FACTORY_REGISTER instead)
  template<typename T>
  static bool register_class(const std::string& id)
  {
    static_assert(std::is_base_of<BaseType, T>::value, "T is not a base of BaseType");

    auto res = get_constructs().emplace(
      std::string(id), 
      std::unique_ptr<construct_base>(new construct<T>())
    );

    if(!res.second)
      throw std::runtime_error(id + " registered multiple times");

    return true;
  }

private:

  /// Base virtual class to store construct<T> as a uniform type in map 
  struct construct_base{
    virtual std::unique_ptr<BaseType> make_instance(Args... args) = 0;
  };
  
  /// Wrapper type to construct type T from the parameters given as template parameters in the factory
  template< typename T >
  struct construct : public construct_base{
    std::unique_ptr<BaseType> make_instance(Args... args)
    {
      return std::make_unique<T>( args... );
    }
  };

  /**
   * Get the id->construct<T> map
   * Using "Meyer's singleton" pattern to avoid static initialization order fiasco
   **/
  static std::map< std::string, std::unique_ptr<construct_base> >& get_constructs()
  {
    static std::map< std::string, std::unique_ptr<construct_base> > res;
    return res;
  }
};

template< typename Factory_t, typename Impl_t >
struct Factory_Registered{
  /**
   * Dummy boolean unique to the registered type for current factory
   * to trigger dynamic initialization for registering
   **/
  static const bool registered;
};

/**
 * Register a type to the factory.
 * Once the type is registered FACTORY_TYPE::make_instance(name, ...) 
 * will create an instance of IMPL_TYPE. 
 * 
 * IMPL_TYPE has to be a complete type when FACTORY_REGISTER is called.
 * 
 * Each FACTORY_REGISTER(Class, name) must have a corresponding 
 * DECLARE_REGISTERED(Class) in RegisterFactory::init(), or implementation may not be fully registered.
 **/
#define FACTORY_REGISTER( FACTORY_TYPE, IMPL_TYPE, name ) template<>\
const bool Factory_Registered<FACTORY_TYPE, IMPL_TYPE>::registered \
            = FACTORY_TYPE::register_class<IMPL_TYPE>( name );

/**
 * Reference an already registered type in the factory in RegisterFactory::init()
 * (this macro won't work outside of RegisterFactory::init())
 * 
 * IMPL_TYPE DOES NOT NEED to be a complete type when FACTORY_REGISTER is called.
 * Only a forward-declaration of the type is necessary. This is useful to avoid 
 * importing the whole header of a registered template class.
 * 
 * Expanation : FACTORY_REGISTER uses dynamic initialization of a static variable to add the class
 * to the id->class map of the factory "before the main". Dynamic initialization (i.e. registering of the class)
 * may not be triggered if the static variable is not imported to the compilation unit where make_instance() is called
 * (this may happen when the static variable is stripped when linking with a static library).
 * To make sure that class is registered before the call to make_instance(), the static variable used to trigger dynamic initialization 
 * must be referenced in the compilation unit where make_instance() is called (using DECLARE_REGISTERED). 
 * Listing all the registered classes in RegisterFactory::init() ensures that all necessary symbols are imported when RegisterFactory::make_instance() is called.
 **/
#define DECLARE_REGISTERED( IMPL_TYPE ) if( !Factory_Registered<RegisteringFactory, IMPL_TYPE>::registered ) {throw std::runtime_error("Class not registered");}
