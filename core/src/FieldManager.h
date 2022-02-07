#pragma once

#include <unordered_map>
#include <set>

#include "kokkos_shared.h"
#include "utils/config/ConfigMap.h"

namespace dyablo {

enum VarIndex {
  ID=0,   /*!< ID Density field index */
  IP=1,   /*!< IP Pressure/Energy field index */
  IE=1,   /*!< IE Energy/Pressure field index */
  IU=2,   /*!< X velocity / momentum index */
  IV=3,   /*!< Y velocity / momentum index */ 
  IW=4,   /*!< Z velocity / momentum index */ 
  IA=5,   /*!< X magnetic field index */ 
  IB=6,   /*!< Y magnetic field index */ 
  IC=7,   /*!< Z magnetic field index */ 
  IBX=5,  /*!< X magnetic field index */ 
  IBY=6,  /*!< Y magnetic field index */ 
  IBZ=7,  /*!< Z magnetic field index */  
  IGPHI=8,/*!< gravitational potential */
  IGX=9,  /*!< X gravitational field index */
  IGY=10,  /*!< Y gravitational field index */
  IGZ=11, /*!< Z gravitational field index */
  IBFX = 0,
  IBFY = 1,
  IBFZ = 2,
  VARINDEX_COUNT=12 /*!< invalid index, just counting number of fields */
};

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
    id2index[id] = _nbfields;
    assert(!field_enabled[id]);
    field_enabled[id] = true;
    _nbfields++;
  }
  std::set<VarIndex> enabled_fields() const
  {
    std::set<VarIndex> res;
    for( int i=0; i<VarIndex::VARINDEX_COUNT; i++ )
      if( field_enabled[i] ) res.insert( (VarIndex)i );
    return res;
  }
  KOKKOS_INLINE_FUNCTION
  int nbfields() const 
  {
    return _nbfields;
  }
  KOKKOS_INLINE_FUNCTION
  int operator[](VarIndex id) const
  {
    assert( field_enabled[id] ); // This variable is not active
    return id2index[id];
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
  FieldManager( const std::set<VarIndex>& active_fields = {} ) 
  {
    for( VarIndex id : active_fields )
    {
      id2index.activate(id);
    }
  }

  id2index_t get_id2index() const
  { 
    return id2index; 
  };

  int nbfields() const
  { 
    return id2index.nbfields(); 
  }

  static FieldManager setup(int ndim, GravityType gravity_type);
  
  static std::string var_name(VarIndex ivar);
  std::set< VarIndex > enabled_fields() const;

  //static const int2str_t& get_id2names_all();
  //static const str2int_t& get_names2id_all();

private:
  id2index_t id2index;
};

} // namespace dyablo 

