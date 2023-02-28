#pragma once

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"
#include "amr/MapUserData.h"
#include "particles/ForeachParticle.h"

namespace dyablo {

class UserData
{
public:
    /***
     * Nested class to access values in fields
     * UserData should not be used directly in Kokkos kernels,
     * create a new FieldAccessor with get*Accessor() to use in kernels
     ***/
    class FieldAccessor; 
    class ParticleAccessor; 
    struct FieldAccessor_FieldInfo;
    using ParticleAccessor_AttributeInfo = FieldAccessor_FieldInfo;

public:
    using FieldView_t = ForeachCell::CellArray_global_ghosted;

    UserData( const UserData& ) = default;
    UserData( UserData&& ) = default;

    UserData( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_cell(foreach_cell), foreach_particle( foreach_cell.get_amr_mesh(), configMap )
    {}

    /***
     * @brief Return a CellArray_global_ghosted::Shape_t instance 
     * with the same size as all fields in current UserData
     * UserData must have at least on active field
     ***/
    const FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_HOST_RELEASE( field_index.size() > 0, "Cannot getShape() of an empty UserData" );
        return fields.getShape();
    }

    void remap( MapUserData& mapUserData )
    {
        FieldView_t fields_old = fields;

        //if( fields.U.extent(2) != foreach_cell.get_amr_mesh().getNumOctants() 
        // || fields.Ughost.extent(2) != foreach_cell.get_amr_mesh().getNumGhosts()  ) 
        {   // AMR mesh was updated : reallocate `max_field_count` fields with right oct count
            // std::cout << "Reallocate : add octs " << fields.U.extent(2) << " -> " << foreach_cell.get_amr_mesh().getNumOctants() << std::endl;
            // std::cout << "Reallocate : add ghosts " << fields.Ughost.extent(2) << " -> " << foreach_cell.get_amr_mesh().getNumGhosts() << std::endl;
            this->fields = foreach_cell.allocate_ghosted_array( "UserData_fields", FieldManager(this->max_field_count) );
        }

        mapUserData.remap( fields_old, fields );
    }

    /**
     * Add new fields with unique identifiers 
     * names should not be already present
     **/
    void new_fields( const std::set<std::string>& names)
    {
        if( this->nbFields() != 0 )
        {
            DYABLO_ASSERT_HOST_RELEASE( fields.U.extent(2) == foreach_cell.get_amr_mesh().getNumOctants(), "UserData internal error : mismatch between allocated size and octant count" );
            DYABLO_ASSERT_HOST_RELEASE( fields.Ughost.extent(2) == foreach_cell.get_amr_mesh().getNumGhosts(), "UserData internal error : mismatch between allocated size and ghost octant count" );
        }
        
        int needed_field_count = nbFields() + names.size();
        this->max_field_count = std::max( this->max_field_count, needed_field_count );
        int allocated_field_count = fields.nbfields();
        if( needed_field_count > allocated_field_count )
        {   // Not enough fields : resize to add fields
            std::cout << "Reallocate : add fields " << allocated_field_count << " -> " << max_field_count << std::endl;
            auto fields_new = foreach_cell.allocate_ghosted_array( "UserData_fields", FieldManager(max_field_count) );
            if( allocated_field_count != 0 )
            {
                Kokkos::deep_copy( 
                    Kokkos::subview(fields_new.U, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                    fields.U
                );
                Kokkos::deep_copy( 
                    Kokkos::subview(fields_new.Ughost, Kokkos::ALL(), std::pair(0,allocated_field_count), Kokkos::ALL() ),
                    fields.Ughost
                );
            }
            fields = fields_new;
        }

        for( const std::string& name : names )
        {
            if( this->has_field(name) )
                throw std::runtime_error(std::string("UserData::new_fields() - field already exists : ") + name);
            /// Find first free ivar in `fields` view
            auto first_free = [&]() -> int
            {
                for(int i=0; i<fields.nbfields(); i++)
                {
                    bool free = true;
                    for( auto& p : field_index )
                    {
                        if(p.second.index == i)
                            free = false;
                    }
                    if( free ) return i;
                }
                DYABLO_ASSERT_HOST_RELEASE(false, "UserData internal error : not enough fields allocated");
                return -1;
            };
            
            int index = first_free();
            field_index[name].index = index;
            const auto& U = fields.U;
            Kokkos::parallel_for( "zero_new_field", U.extent(0)*U.extent(2),
                KOKKOS_LAMBDA( uint32_t i )
            {
                uint32_t iCell = i%U.extent(0);
                uint32_t iOct  = i/U.extent(0);
                U(iCell, index, iOct) = 0;
            });
            const auto& Ughost = fields.Ughost;
            Kokkos::parallel_for( "zero_new_field_ghost", Ughost.extent(0)*Ughost.extent(2),
                KOKKOS_LAMBDA( uint32_t i )
            {
                uint32_t iCell = i%Ughost.extent(0);
                uint32_t iOct  = i/Ughost.extent(0);
                Ughost(iCell, index, iOct) = 0;
            });
        }
    }

    /// Check if field exists
    bool has_field(const std::string& name) const
    {
        return field_index.end() != field_index.find(name);
    }

    std::set<std::string> getEnabledFields() const
    {
        std::set<std::string> res;
        for( const auto& p : field_index )
        {
            res.insert( p.first );
        }
        return res;
    }   

    // Get View associated with field name
    const FieldView_t getField(const std::string& name) const
    {
        if( !this->has_field(name)  )
            throw std::runtime_error(std::string("UserData::getField() - field doesn't exist : ") + name);
        
        int index = field_index.at(name).index;

        auto field = foreach_cell.allocate_ghosted_array( std::string("field_")+name, FieldManager(1) );
        Kokkos::deep_copy( 
            field.U,
            Kokkos::subview(fields.U, Kokkos::ALL(), std::pair(index, index+1) , Kokkos::ALL() )
        );

        return field;
    }

    /// Change field name from src to dest. If dest already exist it is replaced
    void move_field( const std::string& dest, const std::string& src )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_field(src), "UserData::move_field() - field doesn't exist : " << src);

        field_index[ dest ] = field_index.at( src );
        field_index.erase( src );
    }

    void delete_field( const std::string& name )
    {
        field_index.erase( name );
    }

    // TODO exchange ghost for only some fields
    void exchange_ghosts( const GhostCommunicator& ghost_comm ) const
    {
        fields.exchange_ghosts(ghost_comm);
    }

    void exchange_loadbalance( const GhostCommunicator& ghost_comm )
    {
        const FieldManager fm(fields.nbfields());
        auto new_fields = foreach_cell.allocate_ghosted_array( fields.U.label(), fm );
        ghost_comm.exchange_ghosts<2>(fields.U, new_fields.U );
        fields = new_fields;
    }

    /// Get the number of active fields in UserData
    int nbFields() const
    {
        return field_index.size();
    }

    FieldAccessor getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const;

    using ParticleArray_t = ParticleArray;
    using ParticleAttribute_t = ParticleData;

    /// Create a new particle array
    void new_ParticleArray( const std::string& name, uint32_t num_particles )
    {
        if( this->has_ParticleArray(name) )
            throw std::runtime_error(std::string("UserData::new_ParticleArray() - particle array already exists : ") + name);
        std::string view_name = std::string("UserData_Particles_") + name;
        particle_views.emplace( name, ParticleAttributes{ParticleArray_t(name, num_particles)} );
    }

    /// Create a new attribute for particle array `array_name` (must exist)
    void new_ParticleAttribute( const std::string& array_name, const std::string& attribute_name )
    {
        const FieldManager fm_1(1);

        if( this->has_ParticleAttribute(array_name, attribute_name) )
            throw std::runtime_error(std::string("UserData::new_ParticleAttribute() - particle attribute already exists : ") + attribute_name);

        std::string view_name = std::string("UserData_Particles_") + array_name + "_"+ attribute_name;
        
        const ParticleArray_t& array = particle_views.at(array_name).array;
        std::map<std::string, ParticleAttribute_t>& attributes = particle_views.at(array_name).attributes;
        attributes.emplace( attribute_name, ParticleAttribute_t(array, fm_1) );
    }

    /// Check if UserData contains a ParticleArray with this name
    bool has_ParticleArray(const std::string& name) const
    {
        return particle_views.end() != particle_views.find(name);
    }

    // Check is ParticleArray `array_name` (must exist) has an attribute wiuth this name
    bool has_ParticleAttribute(const std::string& array_name, const std::string& attribute_name ) const
    {
        if( !this->has_ParticleArray(array_name) )
            throw std::runtime_error(std::string("UserData::has_ParticleAttribute() - particle array doesn't exist : ") + array_name);

        const auto & attributes = particle_views.at(array_name).attributes;
        return attributes.end() != attributes.find(attribute_name);
    }

    std::set<std::string> getEnabledParticleArrays() const
    {
        std::set<std::string> res;
        for( const auto& p : particle_views )
        {
            res.insert( p.first );
        }
        return res;
    } 

    std::set<std::string> getEnabledParticleAttributes( const std::string& array_name ) const
    {
        if( !this->has_ParticleArray(array_name) )
            throw std::runtime_error(std::string("UserData::getEnabledParticleAttributes() - particle array doesn't exist : ") + array_name);


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
            throw std::runtime_error(std::string("UserData::new_ParticleArray() - particle array doesn't exist : ") + array_name);

        return particle_views.at(array_name).array;
    }
    
    // Get particle attribute with name `attribute_name` from particle array `array_name`
    const ParticleAttribute_t& getParticleAttribute( const std::string& array_name, const std::string& attribute_name ) const
    {
        if( !this->has_ParticleAttribute(array_name, attribute_name) )
            throw std::runtime_error(std::string("UserData::getParticleAttribute() - particle attribute doesn't exist : ") + attribute_name);

        return particle_views.at(array_name).attributes.at(attribute_name);
    }

    void move_ParticleAttribute( const std::string& array_name, const std::string& attr_dest, const std::string& attr_src )
    {
        if( !this->has_ParticleAttribute(array_name, attr_src) )
            throw std::runtime_error(std::string("UserData::move_ParticleAttribute() - particle attribute doesn't exist : ") + attr_src);

        auto & attrs = particle_views.at(array_name).attributes;
        attrs[ attr_dest ] = attrs.at(attr_src);
        attrs.erase(attr_src);        
    }

    void delete_ParticleAttribute(const std::string& array_name, const std::string& attribute_name)
    {
        if( !this->has_ParticleAttribute(array_name, attribute_name) )
            throw std::runtime_error(std::string("UserData::delete_ParticleAttribute() - particle attribute doesn't exist : ") + attribute_name);

        particle_views.at(array_name).attributes.erase(attribute_name);
    }

    ParticleAccessor getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const;
    
    // TODO distribute only some attributes
    void distributeParticles( const std::string& array_name )
    {
        if( !this->has_ParticleArray(array_name) )
            throw std::runtime_error(std::string("UserData::new_ParticleArray() - particle array doesn't exist : ") + array_name);

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
        
        for( int i=0; i<attr_names.size(); i++ )
        {
            this->new_ParticleAttribute(array_name, attr_names[i]);
            part_comm.exchange_ghosts<0>( pdata[i], getParticleAttribute(array_name, attr_names[i]).particle_data );
            pdata[i] = Kokkos::View< real_t**, Kokkos::LayoutLeft >();
        }
    }

private:
    ForeachCell& foreach_cell;
    FieldView_t fields;
    struct field_index_t
    {
        int index;
    };
    std::map<std::string, field_index_t> field_index;
    int max_field_count = 0;

    ForeachParticle foreach_particle;
    struct ParticleAttributes
    {
        ParticleArray_t array;
        std::map<std::string, ParticleAttribute_t> attributes;
    };
    std::map<std::string, ParticleAttributes> particle_views;
};

struct UserData::FieldAccessor_FieldInfo
{
    std::string name; /// Name as in VarIndex.h
    VarIndex id; /// id to use to access with at()
};

class UserData::FieldAccessor
{
public:
    static constexpr int MAX_FIELD_COUNT = 32;
    using FieldInfo = FieldAccessor_FieldInfo;

    FieldAccessor() = default;
    FieldAccessor(const FieldAccessor& ) = default;
    FieldAccessor(FieldAccessor& ) = default;
    FieldAccessor& operator=(const FieldAccessor& ) = default;
    FieldAccessor& operator=(FieldAccessor& ) = default;

    KOKKOS_INLINE_FUNCTION
    int nbFields() const
    {
        return fm_ivar.nbfields();
    }

    FieldAccessor(const UserData& user_data, const std::vector<FieldInfo>& fields_info)
        : fields(user_data.fields)
    {
        DYABLO_ASSERT_HOST_RELEASE( fields_info.size() > 0, "fields_info cannot be empty" );

        int i=0; 
        for( const FieldInfo& info : fields_info )
        {
            // All required fields must have the same size (old/not old)
            int index = user_data.field_index.at(info.name).index;
            fm_ivar.activate( info.id, index );
            fm_active[i] = index; // TODO : maybe reorder?
            i++;
        }
        fields = user_data.fields;
        DYABLO_ASSERT_HOST_RELEASE( fields_info.size() == (size_t)fm_ivar.nbfields(), "fields_info contains duplicate" );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ForeachCell::CellIndex& iCell, const VarIndex& ivar ) const
    {
        return fields.at_ivar( iCell, fm_ivar[ivar] );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachCell::CellIndex& iCell, int ivar ) const
    {
        return fields.at_ivar( iCell, fm_active[ivar] );
    }

    KOKKOS_INLINE_FUNCTION
    FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG(nbFields() > 0, "Cannot getShape() of an empty UserData" );
        return fields.getShape();
    }

protected:
    id2index_t fm_ivar; // ivar from fields_info to position in `fields` view
    Kokkos::Array< int, MAX_FIELD_COUNT > fm_active; // ivar from int sequence to position in `fields` view
    FieldView_t fields;
};

inline UserData::FieldAccessor UserData::getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const
{
    return FieldAccessor(*this, fields_info);
}

class UserData::ParticleAccessor
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

    ParticleAccessor(const UserData& user_data, const std::string& array_name, const std::vector<AttributeInfo>& attr_info)
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

inline UserData::ParticleAccessor UserData::getParticleAccessor( const std::string& array_name, const std::vector<ParticleAccessor_AttributeInfo>& attribute_info ) const
{
    return ParticleAccessor(*this, array_name, attribute_info);
}


}// namespace dyablo