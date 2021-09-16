#include "io_utils.h"

namespace dyablo {

// =======================================================
// =======================================================
int build_var_to_write_map(str2int_t        & map,
			   const HydroParams& params,
			   const ConfigMap  & configMap)
{

  // Read parameter file and get the list of variable field names
  // we wish to write 
  // variable names must be comma-separated (no quotes !)
  // e.g. write_variables=rho,vx,vy,unknown
  std::string write_variables = configMap.getString("output", "write_variables", "rho,rho_vx");

  // second retrieve available / allowed names
  FieldManager fieldMgr = FieldManager::setup(params, configMap);
  id2index_t fm = fieldMgr.get_id2index();
  
  str2int_t avail_names;
  for(VarIndex ivar : fieldMgr.enabled_fields())
    avail_names[ fieldMgr.var_name(ivar) ] = fm[ivar];

  // now tokenize
  std::istringstream iss(write_variables);
  std::string token;
  while (std::getline(iss, token, ',')) {

    // check if token is valid, i.e. present in avail_names
    auto got = avail_names.find(token);
    //std::cout << " " << token << " " << got->second << "\n";
    
    // if token is valid, we insert it into map
    if (got != avail_names.end()) {
      map[token] = got->second;
    }
    
  }
  
  return map.size();
    
} // build_var_to_write_map

} // namespace dyablo
