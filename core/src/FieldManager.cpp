#include "FieldManager.h"

namespace dyablo {

std::string FieldManager::var_name(VarIndex ivar)
{
  static std::unordered_map<VarIndex, std::string> res( var_names().begin(),  var_names().end());
  return res.at(ivar);
}

std::set< VarIndex > FieldManager::enabled_fields() const
{
  return id2index.enabled_fields();
}

} // namespace dyablo
