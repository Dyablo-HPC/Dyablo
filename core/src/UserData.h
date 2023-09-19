#pragma once

#include "UserData_fields.h"
#include "UserData_particles.h"

namespace dyablo {

class UserData
{
public:
    /***
     * Nested class to access values in fields
     * UserData should not be used directly in Kokkos kernels,
     * create a new FieldAccessor with get*Accessor() to use in kernels
     ***/
    using FieldAccessor = UserData_fields::FieldAccessor; 
    using ParticleAccessor = UserData_particles::ParticleAccessor; 
    using FieldAccessor_FieldInfo = UserData_fields::FieldAccessor_FieldInfo;
    using ParticleAccessor_AttributeInfo = UserData_particles::ParticleAccessor_AttributeInfo;

public:
    using FieldView_t = ForeachCell::CellArray_global_ghosted;

    UserData( const UserData& ) = default;
    UserData( UserData&& ) = default;

    UserData( ConfigMap& configMap, ForeachCell& foreach_cell )
    : fields(configMap, foreach_cell), particles(configMap, foreach_cell)
    {}

    /***
     * @brief Return a CellArray_global_ghosted::Shape_t instance 
     * with the same size as all fields in current UserData
     * UserData must have at least on active field
     ***/
    const FieldView_t::Shape_t getShape() const
    {
        return fields.getShape();
    }

    void remap( MapUserData& mapUserData )
    {
        fields.remap(mapUserData);
    }

    /**
     * Add new fields with unique identifiers 
     * names should not be already present
     **/
    void new_fields( const std::set<std::string>& names)
    {
        fields.new_fields(names);
    }

    /// Check if field exists
    bool has_field(const std::string& name) const
    {
        return fields.has_field(name);
    }

    std::set<std::string> getEnabledFields() const
    {
        return fields.getEnabledFields();
    }   

    // Get View associated with field name
    const FieldView_t getField(const std::string& name) const
    {
        return fields.getField(name);
    }

    /// Change field name from src to dest. If dest already exist it is replaced
    void move_field( const std::string& dest, const std::string& src )
    {
        fields.move_field(dest,src);
    }

    void delete_field( const std::string& name )
    {
        fields.delete_field(name);
    }

    // TODO exchange ghost for only some fields
    void exchange_ghosts( const GhostCommunicator& ghost_comm ) const
    {
        fields.exchange_ghosts(ghost_comm);
    }

    void exchange_loadbalance( const GhostCommunicator& ghost_comm )
    {
        fields.exchange_loadbalance(ghost_comm);
    }

    /// Get the number of active fields in UserData
    int nbFields() const
    {
        return fields.nbFields();
    }

    FieldAccessor getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const
    {
        return fields.getAccessor(fields_info);
    }

    using ParticleArray_t = UserData_particles::ParticleArray_t;
    using ParticleAttribute_t = UserData_particles::ParticleAttribute_t;

    /// Create a new particle array
    void new_ParticleArray( const std::string& name, uint32_t num_particles )
    {
        particles.new_ParticleArray(name, num_particles);
    }

    /// Create a new attribute for particle array `array_name` (must exist)
    void new_ParticleAttribute( const std::string& array_name, const std::string& attribute_name )
    {
        particles.new_ParticleAttribute(array_name, attribute_name);
    }

    /// Check if UserData contains a ParticleArray with this name
    bool has_ParticleArray(const std::string& name) const
    {
        return particles.has_ParticleArray(name);
    }

    // Check is ParticleArray `array_name` (must exist) has an attribute wiuth this name
    bool has_ParticleAttribute(const std::string& array_name, const std::string& attribute_name ) const
    {
        return particles.has_ParticleAttribute(array_name, attribute_name);
    }

    /// Get identifiers for all enabled particle arrays
    std::set<std::string> getEnabledParticleArrays() const
    {
        return particles.getEnabledParticleArrays();
    } 

    /// Get identifiers for all enabled attribudes of particle array `array_name`
    std::set<std::string> getEnabledParticleAttributes( const std::string& array_name ) const
    {
        return particles.getEnabledParticleAttributes( array_name );
    } 

    // Get particle array associated with name
    ParticleArray_t getParticleArray( const std::string& array_name ) const
    {
        return particles.getParticleArray(array_name);
    }
    
    // Get particle attribute with name `attribute_name` from particle array `array_name`
    ParticleAttribute_t getParticleAttribute( const std::string& array_name, const std::string& attribute_name ) const
    {
        return particles.getParticleAttribute(array_name, attribute_name);
    }

    /***
     * @brief Change name of particle attribute from `array_name`/`attr_src` to `array_name`/`attr_dest`
     * NOTE : order of parameters is dest, src like in Kokkos deep_copy
     * WARNING : Invalidates all accessors containing source attribute 
     ***/
    void move_ParticleAttribute( const std::string& array_name, const std::string& attr_dest, const std::string& attr_src )
    {
        particles.move_ParticleAttribute(array_name, attr_dest, attr_src);
    }

    /***
     * @brief Delete attribte `array_name`/`attribute_name` from user data
     * WARNING : Invalidates all accessors containing this attribute
     ***/
    void delete_ParticleAttribute(const std::string& array_name, const std::string& attribute_name)
    {
        particles.delete_ParticleAttribute(array_name, attribute_name);
    }

    /***
     * @brief create a ParticleAccessor to access attributes listed in `attribute_info` from particle array `array_name`
     * NOTE : Accessors may be invalidated by some methods from UserData (e.g. deleting or moving an attribute or an array)
     * Do not keep invalidated accessors since live accessors may prevent Kokkos::View deallocation in when deleting or moving user data
     ***/
    ParticleAccessor getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const
    {
        return particles.getParticleAccessor( array_name, attribute_info ); 
    }
    
    /***
     * @brief Distribute position array and attributes for particle array `array_name`
     * WARNING : Invalidates all accessors containing this particle array
     ***/
    void distributeParticles( const std::string& array_name )
    {
        particles.distributeParticles(array_name);
    }

    void distributeAllParticles()
    {
        particles.distributeAllParticles();
    }

private:
    UserData_fields fields;
    UserData_particles particles;
};

}// namespace dyablo