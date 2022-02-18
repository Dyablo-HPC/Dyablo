#include "FieldManager.h"

namespace dyablo {

namespace{
const std::vector< std::pair< VarIndex, std::string > >& var_names()
{
  static std::vector< std::pair< VarIndex, std::string > > res{
    {ID, "rho"},
    {IU, "rho_vx"},
    {IV, "rho_vy"},
    {IW, "rho_vz"},
    {IE, "e_tot"},
    {IA, "bx"},
    {IB, "by"},
    {IC, "bz"},
    {IGX, "igx"},
    {IGY, "igy"},
    {IGZ, "igz"}
  };
  return res;
}
} // namespace

std::string FieldManager::var_name(VarIndex ivar)
{
  static std::unordered_map<VarIndex, std::string> res( var_names().begin(),  var_names().end());
  return res.at(ivar);
}

std::set< VarIndex > FieldManager::enabled_fields() const
{
  return id2index.enabled_fields();
}

FieldManager FieldManager::setup(int ndim, GravityType gravity_type) {    
  
  // always enable rho, energy and velocity components
  std::set< VarIndex > enabled_vars( {ID, IP, IE, IU, IV} );
  
  bool three_d = ndim == 3 ? 1 : 0;

  if( three_d ) enabled_vars.insert( IW );

  if (gravity_type & GRAVITY_FIELD) {
    enabled_vars.insert( IGX );
    enabled_vars.insert( IGY );
    if( three_d ) enabled_vars.insert( IGZ );
  }
  
  return FieldManager( enabled_vars );

}

} // namespace dyablo
