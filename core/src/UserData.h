#pragma once

#include <map>
#include <vector>

#include "utils/config/ConfigMap.h"
#include "foreach_cell/ForeachCell.h"

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
    struct FieldAccessor_FieldInfo;

public:
    using FieldView_t = ForeachCell::CellArray_global_ghosted;

    UserData( const UserData& ) = default;
    UserData( UserData&& ) = default;

    UserData( ConfigMap& configMap, ForeachCell& foreach_cell )
    :   foreach_cell(foreach_cell)
    {}

    /***
     * @brief Return a CellArray_global_ghosted instance 
     * with the same size as all fields in current UserData
     * Note : accessing values stored in result is UB
     * UserData must have at least on active field
     ***/
    const FieldView_t& getShape() const
    {
        DYABLO_ASSERT_HOST_RELEASE( field_views.size() > 0, "Cannot getShape() of an empty UserData" );
        return field_views.begin()->second;
    }

    /**
     * Add new fields with unique identifiers 
     * names should not be already present
     **/
    void new_fields( const std::set<std::string>& names)
    {
        const FieldManager fm_1(1);

        for( const std::string& name : names )
        {
            DYABLO_ASSERT_HOST_RELEASE( !this->has_field(name) , "UserData::new_fields() - field already exists : " << name);
            std::string view_name = std::string("UserData_") + name;
            field_views.emplace( name, foreach_cell.allocate_ghosted_array( view_name, fm_1 ) );
        }
    }

    /// Check if field exists
    bool has_field(const std::string& name) const
    {
        return field_views.end() != field_views.find(name);
    }

    std::set<std::string> getEnabledFields() const
    {
        std::set<std::string> res;
        for( const auto& p : field_views )
        {
            res.insert( p.first );
        }
        return res;
    }   

    /// Get View associated with field name
    const FieldView_t& getField(const std::string& name) const
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_field(name), "UserData::getField() - field doesn't exist : " << name);

        return field_views.at(name);
    }

    /// Change field name from src to dest. If dest already exist it is replaced
    void move_field( const std::string& dest, const std::string& src )
    {
        DYABLO_ASSERT_HOST_RELEASE( this->has_field(src), "UserData::move_field() - field doesn't exist : " << src);

        field_views[ dest ] = field_views.at( src );
        field_views.erase( src );
    }

    void delete_field( const std::string& name )
    {
        field_views.erase( name );
    }

    // TODO exchange ghost for only some fields
    void exchange_ghosts( const GhostCommunicator& ghost_comm ) const
    {
        for( const auto& p : field_views )
        {
            p.second.exchange_ghosts(ghost_comm);
        }
    }

    void exchange_loadbalance( const GhostCommunicator& ghost_comm )
    {
        for( auto& p : field_views )
        {
            const FieldManager fm_1(1);
            auto new_view = foreach_cell.allocate_ghosted_array( p.second.U.label(), fm_1 );
            ghost_comm.exchange_ghosts<2>(p.second.U, new_view.U );
            p.second = new_view;
        }
    }

    /// Get the number of active fields in UserData
    int nbFields() const
    {
        return field_views.size();
    }

    FieldAccessor getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const;
    
private:
    ForeachCell& foreach_cell;
    std::map<std::string, FieldView_t> field_views;
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
        return m_nbFields;
    }

    FieldAccessor(const UserData& user_data, const std::vector<FieldInfo>& fields_info)
     : m_nbFields(fields_info.size())
    {
        for( const FieldInfo& info : fields_info )
        {
            fm.activate( info.id );
            int index = fm[info.id];
            DYABLO_ASSERT_HOST_RELEASE( index < MAX_FIELD_COUNT, "fm index out of bound" );
            field_views[index] = user_data.getField( info.name );
        }
        DYABLO_ASSERT_HOST_RELEASE( fields_info.size() == (size_t)fm.nbfields(), "fields_info contains duplicate" );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ForeachCell::CellIndex& iCell, const VarIndex& ivar ) const
    {
        return field_views[ fm[ivar] ].at_ivar( iCell, 0);
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ForeachCell::CellIndex& iCell, int ivar ) const
    {
        assert( ivar < nbFields() );
        return field_views[ ivar ].at_ivar( iCell, 0);
    }

    KOKKOS_INLINE_FUNCTION
    FieldView_t::Shape_t getShape() const
    {
        assert(nbFields() > 0);
        return field_views[0].getShape();
    }

protected:
    using field_views_t = Kokkos::Array<FieldView_t, MAX_FIELD_COUNT>;

    id2index_t fm;
    field_views_t field_views;
    int m_nbFields;
};

inline UserData::FieldAccessor UserData::getAccessor( const std::vector<FieldAccessor_FieldInfo>& fields_info ) const
{
    return FieldAccessor(*this, fields_info);
}


}// namespace dyablo