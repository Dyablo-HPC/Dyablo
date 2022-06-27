#include "FieldManager.h"

namespace dyablo {

std::string FieldManager::var_name(VarIndex ivar)
{
  static std::unordered_map<VarIndex, std::string> res( var_names().begin(),  var_names().end());
  return res.at(ivar);
}

VarIndex FieldManager::getiVar(const std::string& name)
{
  static std::unordered_map<std::string, VarIndex> res = []()
  {
    std::unordered_map<std::string, VarIndex> res;
    for(const auto& v : var_names() )
    {
      res.insert( std::make_pair(v.second, v.first) );
    }
    return res;
  }();
  return res.at(name);
}

std::set< VarIndex > FieldManager::enabled_fields() const
{
  return id2index.enabled_fields();
}

} // namespace dyablo
