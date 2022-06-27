#pragma once

#include <unordered_map>
#include <set>

#include "kokkos_shared.h"
#include "VarIndex.h"
#include "utils/config/ConfigMap.h"


namespace dyablo {

//! a convenience alias to map id to variable names
using int2str_t = std::unordered_map<int,std::string>;
//! a convenience alias to map variable names to id
using str2int_t = std::unordered_map<std::string,int>;

/**
 * a convenience alias to map id (enum) to index used in DataArray
 **/
class id2index_t{
private:
  Kokkos::Array < int, VarIndex::VARINDEX_COUNT > id2index {};
  Kokkos::Array < bool,VarIndex::VARINDEX_COUNT > field_enabled {};
  int _nbfields = 0;
public:
  void activate( VarIndex id )
  {
    id2index[(int)id] = _nbfields;
    assert(!field_enabled[(int)id]);
    field_enabled[(int)id] = true;
    _nbfields++;
  }
  std::set<VarIndex> enabled_fields() const
  {
    std::set<VarIndex> res;
    for( int i=0; i<(int)VarIndex::VARINDEX_COUNT; i++ )
      if( field_enabled[i] ) res.insert( (VarIndex)i );
    return res;
  }
  KOKKOS_INLINE_FUNCTION
  int nbfields() const 
  {
    return _nbfields;
  }

  KOKKOS_INLINE_FUNCTION
  bool enabled(VarIndex id) const
  {
    return field_enabled[(int)id];
  }

  KOKKOS_INLINE_FUNCTION
  int operator[](VarIndex id) const
  {
    assert( enabled(id) ); // This variable is not active
    return id2index[(int)id];
  }
};

/**
 * Field manager class.
 *
 * Initialize a std::unordered_map object to map enum ComponentIndex 
 * to an actual integer depending on runtime configuration (e.g. is 
 * MHD / magnetic field components valid, etc...).
 */
class FieldManager {  
public:
  /**
   * Create a new FieldManager with specific fields
   * All VarIndex must be created befor this constructor with getiVar()
   **/
  FieldManager( const std::set<VarIndex>& active_fields = {} ) 
  {
    for( VarIndex id : active_fields )
    {
      id2index.activate(id);
    }
  }

  /**
   * Create a new FieldManager with unnamed fields
   * Generated VarIndexes can be fetched using enabled_fields()
   * VarIndexes generated this way should not be used with var_name()
   * This is usually used for temporary arrays when VarIndexes don't need to be conserved between kernels
   **/
  FieldManager( int count ) 
  {
    for( int i=0; i<count; i++ )
    {
      id2index.activate((VarIndex)i);
    }
  }

  id2index_t get_id2index() const
  { 
    return id2index; 
  }

  int nbfields() const
  { 
    return id2index.nbfields(); 
  }
  
  static std::string var_name(VarIndex ivar);
  static VarIndex getiVar(const std::string& name);
  std::set< VarIndex > enabled_fields() const;
private:
  id2index_t id2index;
};

} // namespace dyablo 

