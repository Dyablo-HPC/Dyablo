#pragma once

#include "FieldManager.h"
#include "UserData.h"

namespace dyablo { 

enum VarIndex_legacy{
    ID, IP, IE=IP, IU, IV, IW, IGX, IGY, IGZ
};

class LegacyDataArray : public UserData::FieldAccessor
{
public :
    LegacyDataArray( const UserData& U )
    : UserData::FieldAccessor(U, {
            {"rho", ID},
            {"e_tot", IE},
            {"rho_vx", IU},
            {"rho_vy", IV},
            {"rho_vz", IW},
        })
    {}

    LegacyDataArray( const UserData::FieldAccessor& U )
    : UserData::FieldAccessor(U)
    {}

    KOKKOS_INLINE_FUNCTION
    real_t& operator()( uint32_t iCell, int iVar, uint32_t iOct) const
    {
        assert(iVar<fm.nbfields());
        return this->field_views[iVar].U(iCell, 0, iOct);
    }

    KOKKOS_INLINE_FUNCTION
    const real_t& ghost_val( uint32_t iCell, int iVar, uint32_t iOct) const
    {
        assert(iVar<fm.nbfields());
        return this->field_views[iVar].Ughost(iCell, 0, iOct);
    }
    
    KOKKOS_INLINE_FUNCTION
    id2index_t get_id2index()
    {
        return this->fm;
    }
};

} // namespace dyablo