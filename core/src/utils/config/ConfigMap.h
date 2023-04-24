#pragma once

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <type_traits>

#include "utils/mpi/GlobalMpiSession.h"
#include "utils/config/inih/ini.h"
#include "utils/misc/Dyablo_assert.h"
#include "enums.h"
#include "named_enum.h"
#include <cassert>

namespace Impl{

/**
 * Parse string to type T
 * @tparam Type to convert to
 * @param str string to parse into type T
 * 
 * Specialization when type is not an enum :
 * use std::ostringstream::operator<< to parse string.
 * throws std::runtime_error if there is a parsing error (either << fails or stream contains multiple tokens) 
 **/
template< typename T >
typename std::enable_if< !std::is_enum<T>::value, T >::
type convert_to( const std::string& str )
{
  static_assert( !std::is_same<T, unsigned char>::value && !std::is_same<T, signed char>::value, 
                 "char is not be used as variable type in configMap because its output in an ascii file is confusing" );

  std::istringstream value_stream( str );
  T res;
  value_stream >> res;

  DYABLO_ASSERT_HOST_RELEASE( !value_stream.fail(), "Parse error. typeid : " << typeid(T).name() << ", Input : `" << str << "` -> " << res );
  DYABLO_ASSERT_HOST_RELEASE( value_stream.eof(), "Trailing junk on parse. typeid : " << typeid(T).name() << ", Input : `" << str << "` -> " << res );

  return res;
}

/**
 * convert_to specialization for std::string : 
 * Just return input string
 **/
template<>
inline std::string convert_to<std::string>( const std::string& str )
{
  return str;
}

/**
 * convert_to specialization for bool : 
 * also parse 1/0, yes/no, true/false, on/off as bool
 **/
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
    DYABLO_ASSERT_HOST_RELEASE( false, "Could not parse to boolean. Input : `" << str << "`" );
}

/**
 * convert_to specialization for enums :
 * Parse numbers and names listed in named_enum<T> to enum type T 
 **/
template<typename T>
typename std::enable_if< std::is_enum<T>::value, T >::
type convert_to( const std::string& str )
{
  try { // Try parsing integer
    return static_cast<T>(convert_to<typename std::underlying_type<T>::type>(str));
  } 
  catch(...) {} // Ignore and try other conversion
  
  try { // Try parsing named enum
    return named_enum<T>::from_string(str);
  } 
  catch(...) 
  {
    std::ostringstream sst;
    sst << "Could not parse enum. typeid : " << typeid(T).name() << ", Input : `" << str << "`";
    auto names = named_enum<T>::available_names();
    sst << "Possible values (" << names.size() << ") are :" << std::endl;
    for( const std::string& name : names )
    {
      sst << "- `" << name << "` -> " << named_enum<T>::from_string( name ) << std::endl; 
    }
    DYABLO_ASSERT_HOST_RELEASE( false, sst.str() );
  }
}


/**
 * Convert t to string
 * 
 * Specialization when type is not an enum :
 * use std::istringstream::operator<< to convert.
 **/
template< typename T >
typename std::enable_if< !std::is_enum<T>::value, std::string >::
type to_string( const T& t )
{
  std::ostringstream sst;
  sst << std::boolalpha; // write true/false for bool
  if constexpr( std::is_floating_point_v<T> )
  {
    sst << std::setprecision(std::numeric_limits<T>::max_digits10);
  }
  sst << t;
  return sst.str();
}

/**
 * Convert t to string
 * 
 * Specialization when type an enum :
 * use named_enum<T>::to_string() to convert.
 * throws runtime_error if no name has been defined for t
 **/
template< typename T >
typename std::enable_if< std::is_enum<T>::value, std::string >::
type to_string( const T& t )
{
  try{
    return named_enum<T>::to_string( t ); 
  } catch(...) {
    std::ostringstream sst;
    sst << "Could not find name for enum value. typeid : " << typeid(T).name() << ", Input : `" << t << "`" << std::endl;
    auto names = named_enum<T>::available_names();
    sst << "Possible values (" << names.size() << ") are :" << std::endl;
    for( const std::string& name : names )
    {
      sst << "- `" << name << "` -> " << named_enum<T>::from_string( name ) << std::endl; 
    }
    DYABLO_ASSERT_HOST_RELEASE( false, sst.str() );
  } 
}



} // namespace Impl

class ConfigMap
{
protected:
  /// ConfigMap cannot be copied to ensure default values are registered in original ConfigMap
  ConfigMap( const ConfigMap& ) = default;

public:
  ConfigMap( const std::string& str )
    : ConfigMap(str.c_str())
  {}

  /// Read configmap from string (null terminated)
  ConfigMap(const char* str)
  {
    int error = ini_parse_string(str, valueHandler, this);

    DYABLO_ASSERT_HOST_RELEASE( error == 0, "Error in .ini file line " << error );
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
      DYABLO_ASSERT_HOST_RELEASE(filein, "Could not open .ini file : `" + filename + "`");
      int file_size = filein.tellg();
      
      filein.seekg(0); // rewind
      
      buffer_size = file_size;
      buffer = new char[buffer_size+1];
      
      filein.read(buffer, buffer_size);
      buffer[buffer_size] = '\0';
    }
    
    // broacast buffer size (collective)
    mpi_comm.MPI_Bcast(&buffer_size, 1, 0);

    // all other MPI task need to allocate buffer
    if (myRank>0) {
      //printf("I'm rank %d allocating buffer of size %d\n",myRank,buffer_size);
      buffer = new char[buffer_size+1];
    }

    // broastcast buffer itself (collective)
    mpi_comm.MPI_Bcast(&buffer[0], buffer_size+1, 0);
    
    // now all MPI rank should have buffer filled, try to build a ConfigMap
    ConfigMap configMap(buffer);

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
   * 
   * Most types are parsed using std::ostream::operator<<, but some types are parsed differently:
   * - bool : allowed values are 1/0, yes/no, true/false, on/off 
   * - std::string : are returned without modification until end of line
   * - enumerations : are parsed using named_enum<T>. Names for enum values must be registered in 
   *                  named_enum<T>. Integers can be used for enum values in .ini, but a name must be 
   *                  configured to write default values to last.ini
   * 
   * Note : getValue can be called for different types with the same section/name. 
   * This is not recommended and may have unpredictable beahavior (especially concerning rounding errors)
   **/
  template< typename T >
  T getValue( std::string section, std::string name, const T& default_value )
  {
    bool is_present = hasValue(section, name);
    if( !is_present )
    {
      section = tolower(section);
      name = tolower(name);
      _values[section][name].value = Impl::to_string( default_value );
    }

    T res = getValue<T>( section, name );
    DYABLO_ASSERT_HOST_RELEASE( is_present || default_value == res, 
      "Rounding error when writing defaut value in .ini. default was " << default_value << "but ended up `" << _values[section][name].value << "` in .ini"  );

    return res;
  }
  
  /**
   * Same as getValue(section, name, default) but value must exist
   **/
  template< typename T >
  T getValue( std::string section, std::string name)
  {
    bool is_present = hasValue(section, name);
    DYABLO_ASSERT_HOST_RELEASE( is_present, "Error while parsing .ini " << section << "/" << name << " -- Value not found and no default was provided." )
    
    section = tolower(section);
    name = tolower(name);
    value_container& val = _values[section][name];
    val.used = true;
    T res;
    try {        
      res = Impl::convert_to<T>(val.value);
    }
    catch (const std::exception& e)
    {
      DYABLO_ASSERT_HOST_RELEASE( false, "Error while parsing .ini " << section << "/" << name << " -- " << e.what())
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

    return res;
  }

  bool hasValue( std::string section, std::string name )
  {
    section = tolower(section);
    name = tolower(name);
    return (_values.count(section) != 0) && (_values.at(section).count(name) != 0);
  }

  /**
   * Write configured .ini in o
   * 
   * Variables are printed in the order they are stored in the ConfigMap
   * Comments are added to help understand which variables are defined in .ini file but unused in dyablo, 
   * and which variables have been defaulted because they were not present in original ini file
   * 
   * NOTE : variables present in original .ini files are left as they were defined originally. 
   * ex : enum values set as 1 appear as 1 in last.ini, and will not be replaced by the name in named_enum<T>
   **/
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
protected:
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