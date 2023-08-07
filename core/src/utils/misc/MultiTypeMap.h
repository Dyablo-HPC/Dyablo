#pragma once

#include <string>
#include <sstream>
#include <map>
#include <stdexcept>

namespace dyablo {

template< typename... Ts >
class MultiTypeMap{
private:
  template< typename T >
  using map_t = std::map< std::string, T >;
public:
  MultiTypeMap() = default;

  MultiTypeMap( std::initializer_list< typename map_t<Ts>::value_type > ... is )
   : maps( std::map< std::string, Ts >(is)... )
  {}

  template <typename T> 
  const T& get( const std::string& name ) const
  {
    try{
        return getMap<T>().at(name);
    }
    catch (std::out_of_range& a)
    {
        std::ostringstream out;
        out << "Could not get variable `" << name << "` for this type (" << typeid( T ).name() <<  ") in MultiTypeMap." << std::endl;
        out << "Available variables are : " << std::endl;

        auto print_variables = [&](auto map)
        {
            out << "type : " << typeid( typename decltype(map)::mapped_type ).name() << std::endl;
            for(const auto& p : map)
            {
                out << "  - `" << p.first << "` = " << p.second << std::endl;
            }
        };

        (( print_variables( getMap<Ts>() ) , ... ));

        throw std::runtime_error( out.str() );
    }    
  }

  template <typename T> 
  T& get( const std::string& name )
  {
    const T& res_const = static_cast< const MultiTypeMap* >(this)->get<T>(name);
    return const_cast<T&>( res_const );
  }

  template <typename T> 
  void set( const std::string& name, const T& v )
  {
    getMap<T>()[name] = v;
  }

  template <typename T, typename Func>
  void foreach_var_t(const Func& f)
  {
    for( const auto& p : getMap<T>()  )
    {
      f( p.first, p.second );
    }
  }

  template <typename Func>
  void foreach_var(const Func& f)
  {
    ( foreach_var_t<Ts>(f), ... );
  }

  void print()
  {
    std::cout << "scalar_data : ";

    foreach_var( [](const std::string& name, auto val)
    {
      std::cout << name << "=" << val << " ";
    } );
    
    std::cout << std::endl;
  }

private:
  std::tuple< map_t<Ts>... > maps;
  
  template< typename T >
  map_t<T>& getMap()
  {
    static_assert( (std::is_same_v<T, Ts> || ...), "Type not included in MultiTypeMap type list" );
    return std::get<map_t<T>>(maps);
  }

  template< typename T >
  const map_t<T>& getMap() const
  {
    static_assert( (std::is_same_v<T, Ts> || ...), "Type not included in MultiTypeMap type list" );
    return std::get<map_t<T>>(maps);
  }
};

} // namespace dyablo