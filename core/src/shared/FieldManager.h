/**
 * \file FieldManager.h
 * \brief Define class FieldManager
 *
 * \date January, 21st 2019
 */
#ifndef FIELD_MANAGER_H_
#define FIELD_MANAGER_H_

#include <unordered_map>

#include "enums.h"
#include "HydroParams.h"

//! a convenience alias to map variable names to id
using str2int_t = std::unordered_map<std::string,int>;

//! a convenience alias to map id to variable names
using int2str_t = std::unordered_map<int,std::string>;

/**
 * a convenience alias to map id (enum) to index used in DataArray
 *
 * To clarify again:
 * - id is an enum value
 * - index is an integer between 0 and scalarFieldNb-1, 
 *         used to access a DataArray
 */
using id2index_t = Kokkos::Array<int, COMPONENT_SIZE>;

/**
 * Field manager class.
 *
 * Initialize a std::unordered_map object to map enum ComponentIndex 
 * to an actual integer depending on runtime configuration (e.g. is 
 * MHD / magnetic field components valid, etc...).
 */
class FieldManager {

public:
  FieldManager() {};
  ~FieldManager() {};

  //! number of scalar field, will be used in allocating DataArray
  int numScalarField = 0;

private:
  /**
   * a map containing the list of scalar field available depending on
   * the physics enabled.
   */
  int2str_t index2names;

  /**
   * a map containing the list of scalar field available depending on
   * the physics enabled.
   */
  str2int_t names2index;

  /**
   * a Kokkos::Array to map a ComponentIndex to the actual index used in
   * DataArray.
   */
  id2index_t id2index;

public:
  /**
   * Initialize the Field Manager data (id2names and names2id)
   * using the configMap information.
   */
  void setup(const HydroParams& params, const ConfigMap& configMap) {
    
    const int nbComponent = COMPONENT_SIZE;
    
    /*
     * step1 : build a list of enabled variables, default is all false
     */
    std::array<int,nbComponent> var_enabled;

    for (int i=0; i<nbComponent; ++i)
      var_enabled[i] = 0;

    // always enable rho, energy and velocity components
    bool three_d = params.dimType == 3 ? 1 : 0;

    var_enabled[ID] = 1;
    var_enabled[IP] = 1; // remind that IE and IP aliases 
    var_enabled[IE] = 1;
    var_enabled[IU] = 1;
    var_enabled[IV] = 1;
    var_enabled[IW] = three_d;

    if (params.gravity_type & GRAVITY_FIELD) {
      var_enabled[IGX] = 1;
      var_enabled[IGY] = 1;
      var_enabled[IGZ] = three_d;
    }
    // if (params.mhd_enabled) {
    //   var_enabled[IA] = 1;
    //   var_enabled[IB] = 1;
    //   var_enabled[IC] = 1;
    // }
      
    /*
     * step2 : fill id2index map array.
     */
    
    // build a list of index mapped to a Component Index
    // if variabled is enabled, a unique index is attributed
    // else invalid index -1 is given
    int count = 0;
    for (int id=0; id<nbComponent; ++id) {

      if (var_enabled[id] == 1) {
        id2index[id] = count;
        count++;
      } else {
        id2index[id] = -1;
      }
      
    } // end for

    numScalarField = count;
    
    /* 
     * init indexd2names (numScalarField elements),
     * unordered map of runtime available scalar field
     */
    int2str_t id2namesAll  = get_id2names_all();

    for (int id=0; id<nbComponent; ++id) {
      if (var_enabled[id] == 1) {
        // insert couple  ( index(id), name )
        index2names[ id2index[id] ] = id2namesAll[id];
      }
    }

    /* 
     * init names2index (numScalarField elements),
     * unordered map of runtime available scalar field
     */
    str2int_t names2dAll = get_names2id_all();

    for (int id=0; id<nbComponent; ++id) {
      if (var_enabled[id] == 1) {
        // insert couple  ( index(id), name )
        names2index[ id2namesAll[id] ] = id2index[id];
      }
    }
    
  } // setup
  
  int2str_t get_index2names() { return index2names; }

  str2int_t get_names2index() { return names2index; }

  id2index_t get_id2index() { return id2index; };

  int nbfields() { return numScalarField; }

  /**
   * Builds an unordered_map between enum ComponentIndex and names (string) 
   * using all available fields.
   *
   * \return map of id to names
   */
  static int2str_t get_id2names_all()
  {

    int2str_t map;

    // insert some fields
    map[ID]   = "rho";
    map[IU]   = "rho_vx";
    map[IV]   = "rho_vy";
    map[IW]   = "rho_vz";
    map[IE]   = "e_tot";
    map[IA]   = "bx";
    map[IB]   = "by";
    map[IC]   = "bz";
    map[IGX]  = "igx";
    map[IGY]  = "igy";
    map[IGZ]  = "igz";
    
    return map;
    
  } // get_id2names_all

  /**
   * Builds an unordered_map between fields names (string) and enum 
   * ComponentIndex id, using all available fields.
   *
   * \return map of names to id
   */
  static str2int_t get_names2id_all()
  {
    
    str2int_t map;

    // insert some fields
    map["rho"]      = ID;
    map["rho_vx"]   = IU;
    map["rho_vy"]   = IV;
    map["rho_vz"]   = IW;
    map["e_tot"]    = IE;
    map["bx"]       = IA;
    map["by"]       = IB;
    map["bz"]       = IC;
    map["igx"]      = IGX;
    map["igy"]      = IGY;
    map["igz"]      = IGZ;
    
    return map;
    
  } // get_names2id_all

}; // FieldManager

#endif // FIELD_MANAGER_H_
