#pragma once

#include <string>
#include <map>
#include <vector>

template< typename T >
class named_enum
{
public:
    static const T& from_string( const std::string& str )
    {
        return singleton().name_to_val.at(str);
    }
    static std::string to_string( const T& t )
    {
        return singleton().val_to_name.at(t);
    }
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