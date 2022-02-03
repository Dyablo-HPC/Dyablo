#pragma once

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <type_traits>

#include "utils/mpi/GlobalMpiSession.h"
#include "utils/config/inih/ini.h"
#include "enums.h"
#include <cassert>

namespace Impl{

// TODO move to cpp file
template< typename T >
typename std::enable_if< !std::is_enum<T>::value, T >::
type convert_to( const std::string& str )
{
  static_assert( !std::is_same<T, unsigned char>::value && !std::is_same<T, signed char>::value, 
                 "char is not be used as variable type in configMap because its output in an ascii file is confusing" );

  std::istringstream value_stream( str );
  T res;
  value_stream >> res;

  if( value_stream.fail() ) 
  {
    std::ostringstream sst;
    sst << "Parse error. typeid : " << typeid(T).name() << ", Input : `" << str << "` -> " << res;
    throw std::runtime_error( sst.str() );
  }
  if( !value_stream.eof() ) 
  {
    std::ostringstream sst;
    sst << "Trailing junk on parse. typeid : " << typeid(T).name() << ", Input : `" << str << "` -> " << res;
    throw std::runtime_error( sst.str() );
  }

  return res;
}

template<>
inline std::string convert_to<std::string>( const std::string& str )
{
  return str;
}

template<>
inline bool convert_to<bool>( const std::string& str )
{
  if (!str.compare("1") or 
      !str.compare("yes") or 
      !str.compare("true") or
      !str.compare("on"))
    return true;
  else if (!str.compare("0") or 
      !str.compare("no") or 
      !str.compare("false") or
      !str.compare("off"))
    return false;
  else
    throw std::runtime_error( std::string("Could not parse to boolean. Input : `") + str + "`" );
}

/**
 * convert_to for enums
 * TODO : enable string parse for enums
 **/
template<typename T>
typename std::enable_if< std::is_enum<T>::value, T >::
type convert_to( const std::string& str )
{
  return static_cast<T>(convert_to<typename std::underlying_type<T>::type>(str));
}


} // namespace Impl

class ConfigMap
{
private:
  /// ConfigMap cannot be copied to ensure default values are registered in original ConfigMap
  ConfigMap( const ConfigMap& ) = default;

public:
  /// Read configmap from buffer
  ConfigMap(char* &buffer, int buffer_size)
  {
    int error = ini_parse_buffer(buffer, buffer_size, valueHandler, this);
    if( error != 0 ) throw std::runtime_error(std::string("Error in .ini file line ") + std::to_string(error));
  
    this->print_config = this->getValue<bool>( "ini", "print_config", false );
  }
  ConfigMap( ConfigMap&& ) = default;

  
  
  /// Read configmap from file and broadcast : 
  ///   this is an MPI collective that create the same ConfigMap on every process
  static ConfigMap broadcast_parameters(std::string filename)
  {
    const dyablo::MpiComm& mpi_comm = dyablo::GlobalMpiSession::get_comm_world();
    int myRank = mpi_comm.MPI_Comm_rank();
    
    char* buffer = nullptr;
    int buffer_size = 0;
    
    // MPI rank 0 reads parameter file
    if (myRank == 0) {
      
      // open file and go to the end to get file size in bytes
      std::ifstream filein(filename.c_str(), std::ifstream::ate);
      if( !filein )
        throw std::runtime_error("Could not open .ini file : `" + filename + "`");
      int file_size = filein.tellg();
      
      filein.seekg(0); // rewind
      
      buffer_size = file_size;
      buffer = new char[buffer_size];
      
      filein.read(buffer, buffer_size);
    }
    
    // broacast buffer size (collective)
    mpi_comm.MPI_Bcast(&buffer_size, 1, 0);

    // all other MPI task need to allocate buffer
    if (myRank>0) {
      //printf("I'm rank %d allocating buffer of size %d\n",myRank,buffer_size);
      buffer = new char[buffer_size];
    }

    // broastcast buffer itself (collective)
    mpi_comm.MPI_Bcast(&buffer[0], buffer_size, 0);
    
    // now all MPI rank should have buffer filled, try to build a ConfigMap
    ConfigMap configMap(buffer,buffer_size);

    if (buffer)
      delete [] buffer;

    return configMap;
      
  } // broadcast_parameters

  /**
   * Read the value from configMap corresponding to 
   * ```
   * [section]
   * name=<value>
   * ```
   * Value is then parsed to type T and returned. 
   * Case is ignored in section and name.
   * If parameter is not present, it is added to the configmap
   * Throws std::runtime_error if value cannot be parsed to type T 
   * or if there is still something left in <value> after parsing
   * ( e.g if "2.5" is parsed as integer, ".5" will remain )
   **/
  template< typename T >
  T getValue( std::string section, std::string name, const T& default_value )
  {
    section = tolower(section);
    name = tolower(name);
    bool is_present = (_values.count(section) != 0) && (_values.at(section).count(name) != 0);
    value_container& val = _values[section][name];
    
    if(!is_present)
    {
      std::ostringstream sst;
      sst << std::boolalpha;
      sst << default_value;
      val.value = sst.str();
    }

    val.used = true;
    T res;
    try {        
      res = Impl::convert_to<T>(val.value);
    }
    catch (const std::exception& e)
    {
      throw std::runtime_error( std::string("Error while parsing .ini ") + section + "/" + name + " -- " + e.what() );
    }

    if( print_config )
    {
      std::cout << ".ini ";
      if( is_present ) 
        std::cout << " found   : ";
      else            
        std::cout << " default : ";
      std::cout << std::left << std::boolalpha;
      std::cout << std::setw( 25 ) << section;
      std::cout << std::setw( 25 ) << name;
      std::cout << std::setw( 25 ) << std::string("'")+val.value+"'" ;
      std::cout << " -> " << res;
      std::cout << std::endl;
    }

    assert( is_present || default_value == res );
    return res;
  }


  void output(std::ostream& o)
  {
    constexpr std::string::size_type name_width = 20;
    constexpr std::string::size_type value_width = 20;
    auto initial_format = o.flags();
    o << std::left;
    for( auto p_section : _values )
    {
      const std::string& section_name = p_section.first;
      const std::map<std::string, value_container>& map_section = p_section.second;

      o << "[" << section_name << "]" << std::endl;
      for( auto p_var : map_section )
      {
        const std::string& var_name = p_var.first;
        const value_container& val = p_var.second;

        o << std::setw(std::max(var_name.length(),name_width)) << var_name << " = " << std::setw(std::max(val.value.length(), value_width)) << val.value;
        if( !val.used )
          o << " ; Unused";
        if( !val.from_file )
          o << " ; Default";
        o << std::endl; 
      }
    }
    o.flags(initial_format);
  }
private:
  struct value_container{
    std::string value;
    bool from_file = false;
    bool used = false;
  };

  bool print_config;

  std::map<std::string, std::map< std::string, value_container> > _values;
  static int valueHandler( void* user, const char* section_cstr, const char* name_cstr,
                            const char* value_cstr )
  {
    ConfigMap* reader = (ConfigMap*)user;
    std::string section = tolower(section_cstr);
    std::string name = tolower(name_cstr);
    std::string value (value_cstr);
    reader->_values[section][name] = value_container{std::string(value), true};
    return 1;
  }

  static std::string tolower(std::string str)
  {
    std::transform( str.begin(), str.end(), str.begin(), ::tolower );
    return str;
  }
};