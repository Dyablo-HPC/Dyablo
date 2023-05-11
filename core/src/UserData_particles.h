#pragma once

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"
#include "amr/MapUserData.h"
#include "particles/ForeachParticle.h"

namespace dyablo {

class UserData_particles
{
public:
    /***
     * Nested class to access values in fields
     * UserData_particles should not be used directly in Kokkos kernels,
     * create a new FieldAccessor with get*Accessor() to use in kernels
     ***/
    class ParticleAccessor; 
    struct ParticleAccessor_AttributeInfo;
    

public:
    UserData_particles( const UserData_particles& ) = default;
    UserData_particles( UserData_particles&& ) = default;

    UserData_particles( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_particle( foreach_cell.get_amr_mesh(), configMap )
    {}

    using ParticleArray_t = ParticleArray;
    using ParticleAttribute_t = ParticleData;

    /// Create a new particle array
    void new_ParticleArray( const std::string& name, uint32_t num_particles )
    {
        if( this->has_ParticleArray(name) )
            throw std::runtime_error(std::string("UserData_particles::new_ParticleArray() - particle array already exists : ") + name);
        std::string view_name = std::string("UserData_Particles_") + name;
        particle_views.emplace( name, ParticleAttributes{ParticleArray_t(name, num_particles)} );
    }

    /// Create a new attribute for particle array `array_name` (must exist)
    void new_ParticleAttribute( const std::string& array_name, const std::string& attribute_name )
    {
        const FieldManager fm_1(1);

        if( this->has_ParticleAttribute(array_name, attribute_name) )
            throw std::runtime_error(std::string("UserData_particles::new_ParticleAttribute() - particle attribute already exists : ") + attribute_name);

        std::string view_name = std::string("UserData_Particles_") + array_name + "_"+ attribute_name;
        
        const ParticleArray_t& array = particle_views.at(array_name).array;
        std::map<std::string, ParticleAttribute_t>& attributes = particle_views.at(array_name).attributes;
        attributes.emplace( attribute_name, ParticleAttribute_t(array, fm_1) );
    }

    /// Check if UserData_particles contains a ParticleArray with this name
    bool has_ParticleArray(const std::string& name) const
    {
        return particle_views.end() != particle_views.find(name);
    }

    // Check is ParticleArray `array_name` (must exist) has an attribute wiuth this name
    bool has_ParticleAttribute(const std::string& array_name, const std::string& attribute_name ) const
    {
        if( !this->has_ParticleArray(array_name) )
            throw std::runtime_error(std::string("UserData_particles::has_ParticleAttribute() - particle array doesn't exist : ") + array_name);

        const auto & attributes = particle_views.at(array_name).attributes;
        return attributes.end() != attributes.find(attribute_name);
    }

    /// Get identifiers for all enabled particle arrays
    std::set<std::string> getEnabledParticleArrays() const
    {
        std::set<std::string> res;
        for( const auto& p : particle_views )
        {
            res.insert( p.first );
        }
        return res;
    } 

    /// Get identifiers for all enabled attribudes of particle array `array_name`
    std::set<std::string> getEnabledParticleAttributes( const std::string& array_name ) const
    {
        if( !this->has_ParticleArray(array_name) )
            throw std::runtime_error(std::string("UserData_particles::getEnabledParticleAttributes() - particle array doesn't exist : ") + array_name);


        std::set<std::string> res;
        for( const auto& p : particle_views.at(array_name).attributes )
        {
            res.insert( p.first );
        }
        return res;
    } 

    // Get particle array associated with name
    const ParticleArray_t& getParticleArray( const std::string& array_name ) const
    {
        if( !this->has_ParticleArray(array_name) )
            throw std::runtime_error(std::string("UserData_particles::new_ParticleArray() - particle array doesn't exist : ") + array_name);

        return particle_views.at(array_name).array;
    }
    
    // Get particle attribute with name `attribute_name` from particle array `array_name`
    const ParticleAttribute_t& getParticleAttribute( const std::string& array_name, const std::string& attribute_name ) const
    {
        if( !this->has_ParticleAttribute(array_name, attribute_name) )
            throw std::runtime_error(std::string("UserData_particles::getParticleAttribute() - particle attribute doesn't exist : ") + attribute_name);

        return particle_views.at(array_name).attributes.at(attribute_name);
    }

    /***
     * @brief Change name of particle attribute from `array_name`/`attr_src` to `array_name`/`attr_dest`
     * NOTE : order of parameters is dest, src like in Kokkos deep_copy
     * WARNING : Invalidates all accessors containing source attribute 
     ***/
    void move_ParticleAttribute( const std::string& array_name, const std::string& attr_dest, const std::string& attr_src )
    {
        if( !this->has_ParticleAttribute(array_name, attr_src) )
            throw std::runtime_error(std::string("UserData_particles::move_ParticleAttribute() - particle attribute doesn't exist : ") + attr_src);

        auto & attrs = particle_views.at(array_name).attributes;
        attrs[ attr_dest ] = attrs.at(attr_src);
        attrs.erase(attr_src);    
    }

    /***
     * @brief Delete attribte `array_name`/`attribute_name` from user data
     * WARNING : Invalidates all accessors containing this attribute
     ***/
    void delete_ParticleAttribute(const std::string& array_name, const std::string& attribute_name)
    {
        if( !this->has_ParticleAttribute(array_name, attribute_name) )
            throw std::runtime_error(std::string("UserData_particles::delete_ParticleAttribute() - particle attribute doesn't exist : ") + attribute_name);

        particle_views.at(array_name).attributes.erase(attribute_name);
    }

    /***
     * @brief create a ParticleAccessor to access attributes listed in `attribute_info` from particle array `array_name`
     * NOTE : Accessors may be invalidated by some methods from UserData_particles (e.g. deleting or moving an attribute or an array)
     * Do not keep invalidated accessors since live accessors may prevent Kokkos::View deallocation in when deleting or moving user data
     ***/
    ParticleAccessor getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const;
    
    /***
     * @brief Distribute position array and attributes for particle array `array_name`
     * WARNING : Invalidates all accessors containing this particle array
     ***/
    void distributeParticles( const std::string& array_name )
    {
        if( !this->has_ParticleArray(array_name) )
            throw std::runtime_error(std::string("UserData_particles::new_ParticleArray() - particle array doesn't exist : ") + array_name);

        ParticleArray_t& particle_array = particle_views.at(array_name).array;

        auto attr_names_aux = this->getEnabledParticleAttributes(array_name);
        std::vector<std::string> attr_names(attr_names_aux.begin(), attr_names_aux.end());
        std::vector<Kokkos::View< real_t**, Kokkos::LayoutLeft >> pdata; 
        for( const std::string& attr : attr_names )
        {
            pdata.push_back( this->getParticleAttribute( array_name, attr ).particle_data );
            this->delete_ParticleAttribute(array_name, attr);
        }
        
        GhostCommunicator_kokkos part_comm = foreach_particle.get_distribute_communicator( particle_array );
        uint32_t nbParticles_new = part_comm.getNumGhosts();
        
        ParticleArray particle_array_new( array_name, nbParticles_new );
        part_comm.exchange_ghosts<0>( particle_array.particle_position, particle_array_new.particle_position );

        particle_array = particle_array_new;
        
        for( size_t i=0; i<attr_names.size(); i++ )
        {
            this->new_ParticleAttribute(array_name, attr_names[i]);
            part_comm.exchange_ghosts<0>( pdata[i], getParticleAttribute(array_name, attr_names[i]).particle_data );
            pdata[i] = Kokkos::View< real_t**, Kokkos::LayoutLeft >();
        }
    }

private:
    ForeachParticle foreach_particle;
    struct ParticleAttributes
    {
        ParticleArray_t array;
        std::map<std::string, ParticleAttribute_t> attributes;
    };
    std::map<std::string, ParticleAttributes> particle_views;
};

struct UserData_particles::ParticleAccessor_AttributeInfo
{
    std::string name; /// Name as in VarIndex.h
    VarIndex id; /// id to use to access with at()
};

class UserData_particles::ParticleAccessor
{
public:
    static constexpr int MAX_ATTR_COUNT = 32;
    using AttributeInfo = ParticleAccessor_AttributeInfo;

    ParticleAccessor() = default;
    ParticleAccessor(const ParticleAccessor& ) = default;
    ParticleAccessor(ParticleAccessor& ) = default;
    ParticleAccessor& operator=(const ParticleAccessor& ) = default;
    ParticleAccessor& operator=(ParticleAccessor& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return m_nbFields;
    }

    ParticleAccessor(const UserData_particles& user_data, const std::string& array_name, const std::vector<AttributeInfo>& attr_info)
     : m_nbFields(attr_info.size())
    {
        for( const AttributeInfo& info : attr_info )
        {
            fm.activate( info.id );
            int index = fm[info.id];
            assert( index < MAX_ATTR_COUNT );
            particle_views[index] = user_data.getParticleAttribute( array_name, info.name );
        }
        assert( attr_info.size() == (size_t)fm.nbfields() ); // attr_info contains duplicate
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ForeachParticle::ParticleIndex& iPart, const VarIndex& ivar ) const
    {
        return particle_views[ fm[ivar] ].at_ivar( iPart, 0);
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachParticle::ParticleIndex& iPart, int ivar ) const
    {
        assert( ivar < nbFields() );
        return particle_views[ ivar ].at_ivar( iPart, 0);
    }

    KOKKOS_INLINE_FUNCTION
    ParticleArray_t getShape() const
    {
        assert(nbFields() > 0);
        return particle_views[0];
    }

protected:
    using particle_views_t = Kokkos::Array<ParticleAttribute_t, MAX_ATTR_COUNT>;

    id2index_t fm;
    particle_views_t particle_views;
    int m_nbFields;
};

inline UserData_particles::ParticleAccessor UserData_particles::getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const
{
    return ParticleAccessor(*this, array_name, attribute_info);
}


}// namespace dyablo