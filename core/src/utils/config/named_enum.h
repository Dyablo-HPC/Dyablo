#pragma once

#include <string>
#include <map>
#include <vector>

/**
 * Class to associate enum values to names
 * 
 * Names are configured by specializing named_enum::names
 **/
template< typename T >
class named_enum
{
public:
    /// Parse value from name `str`
    static const T& from_string( const std::string& str )
    {
        return singleton().name_to_val.at(str);
    }
    /// Get name associated to `t`
    static std::string to_string( const T& t )
    {
        return singleton().val_to_name.at(t);
    }
    /// Get list of available names for this enum type
    static std::vector<std::string> available_names()
    {
        std::vector<std::string> names;
        for( const auto& p : singleton().name_to_val )
        {
            names.push_back( p.first );
        }
        return names;
    }

    using init_list = std::vector< std::pair< T, std::string > >;
protected:
    /**
     * List of pair value:name to use to construct singleton
     * A specialization have to be declared by the user for every enum 
     * used with named_enum (declare inline just below the enum declaration)
     **/
    static init_list names;
    static named_enum& singleton()
    {
        // Initialize static variable once with lambda
        static named_enum res = []()
        {
            named_enum res;
            // Add names listed in `names` to internal maps
            for( const auto& p : names )
            {
                const T& t = p.first; 
                const std::string& name = p.second; 
                res.name_to_val.insert({name, t});
                res.val_to_name.insert({t, name});
            }            
            return res;
        }();
        return res;
    }
    std::map<std::string, T> name_to_val;
    std::map<T, std::string> val_to_name;
};