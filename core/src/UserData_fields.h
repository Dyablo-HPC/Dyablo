#pragma once

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"
#include "amr/MapUserData.h"
#include "particles/ForeachParticle.h"

namespace dyablo {

class UserData_fields
{
public:
    /***
     * Nested class to access values in fields
     * UserData_fields should not be used directly in Kokkos kernels,
     * create a new FieldAccessor with get*Accessor() to use in kernels
     ***/
    class FieldAccessor; 
    struct FieldAccessor_FieldInfo;

public:
    using FieldView_t = ForeachCell::CellArray_global_ghosted;

    UserData_fields( const UserData_fields& ) = default;
    UserData_fields( UserData_fields&& ) = default;

    UserData_fields( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_cell(foreach_cell)
    {}

    /***
     * @brief Return a CellArray_global_ghosted::Shape_t instance 
     * with the same size as all fields in current UserData_fields
     * UserData_fields must have at least on active field
     ***/
    const FieldView_t::Shape_t getShape() const
    {
        DYABLO_ASSERT_HOST_RELEASE( field_index.size() > 0, "Cannot getShape() of an empty UserData_fields" );
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
            DYABLO_ASSERT_HOST_RELEASE( fields.U.extent(2) == foreach_cell.get_amr_mesh().getNumOctants(), "UserData_fields internal error : mismatch between allocated size and octant count" );
            DYABLO_ASSERT_HOST_RELEASE( fields.Ughost.extent(2) == foreach_cell.get_amr_mesh().getNumGhosts(), "UserData_fields internal error : mismatch between allocated size and ghost octant count" );
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
                throw std::runtime_error(std::string("UserData_fields::new_fields() - field already exists : ") + name);
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
                DYABLO_ASSERT_HOST_RELEASE(false, "UserData_fields internal error : not enough fields allocated");
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
            throw std::runtime_error(std::string("UserData_fields::getField() - field doesn't exist : ") + name);
        
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
        DYABLO_ASSERT_HOST_RELEASE( this->has_field(src), "UserData_fields::move_field() - field doesn't exist : " << src);

        field_index[ dest ] = field_index.at( src );
        field_index.erase( src );
    }

    void delete_field( const std::string& name )
    {
        field_index.erase( name );
    }

    void exchange_loadbalance( const ViewCommunicator& ghost_comm )
    {
        const FieldManager fm(fields.nbfields());
        auto new_fields = foreach_cell.allocate_ghosted_array( fields.U.label(), fm );
        ghost_comm.exchange_ghosts<2>(fields.U, new_fields.U );
        fields = new_fields;
    }

    /// Get the number of active fields in UserData_fields
    int nbFields() const
    {
        return field_index.size();
    }

    FieldAccessor getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const;

private:
    ForeachCell& foreach_cell;
    FieldView_t fields;
    struct field_index_t
    {
        int index;
    };
    std::map<std::string, field_index_t> field_index;
    int max_field_count = 0;
};

struct UserData_fields::FieldAccessor_FieldInfo
{
    std::string name; /// Name as in VarIndex.h
    VarIndex id; /// id to use to access with at()
};

class GhostCommunicator_full_blocks;

class UserData_fields::FieldAccessor
{
friend GhostCommunicator_full_blocks;
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

    FieldAccessor(const UserData_fields& user_data, const std::vector<FieldInfo>& fields_info)
        : fields(user_data.fields)
    {
        DYABLO_ASSERT_HOST_RELEASE( fields_info.size() > 0, "fields_info cannot be empty" );

        auto unknown_field_error = [&](std::string field_name)
        {
            std::stringstream s;
            s << "Could not find field '" << field_name << "' in UserData" << std::endl;
            s << "Available fields are :" << std::endl;
            for( auto& p : user_data.field_index )
            {
                s << " - '" << p.first << "'" << std::endl;
            }
            return s.str();
        };

        int i=0; 
        for( const FieldInfo& info : fields_info )
        {
            DYABLO_ASSERT_HOST_RELEASE( user_data.has_field(info.name), 
                                        unknown_field_error(info.name) );
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
        DYABLO_ASSERT_KOKKOS_DEBUG(nbFields() > 0, "Cannot getShape() of an empty UserData_fields" );
        return fields.getShape();
    }

protected:
    id2index_t fm_ivar; // ivar from fields_info to position in `fields` view
    Kokkos::Array< int, MAX_FIELD_COUNT > fm_active; // ivar from int sequence to position in `fields` view
    FieldView_t fields;
};

inline UserData_fields::FieldAccessor UserData_fields::getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const
{
    return FieldAccessor(*this, fields_info);
}


}// namespace dyablo