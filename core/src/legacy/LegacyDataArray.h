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
    : UserData::FieldAccessor(U.getAccessor( {
            {"rho", ID},
            {"e_tot", IE},
            {"rho_vx", IU},
            {"rho_vy", IV},
            {"rho_vz", IW},
        }))
    {}

    LegacyDataArray( const UserData::FieldAccessor& U )
    : UserData::FieldAccessor(U)
    {}

    KOKKOS_INLINE_FUNCTION
    real_t& operator()( uint32_t iCell, int iVar, uint32_t iOct) const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG(iVar<fm_ivar.nbfields(), "iVar out of bounds");
        return this->fields.U(iCell, this->fm_active[iVar], iOct);
    }

    KOKKOS_INLINE_FUNCTION
    const real_t& ghost_val( uint32_t iCell, int iVar, uint32_t iOct) const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG(iVar<fm_ivar.nbfields(), "iVar out of bounds");
        return this->fields.Ughost(iCell, this->fm_active[iVar], iOct);
    }
    
    KOKKOS_INLINE_FUNCTION
    id2index_t get_id2index()
    {
        return FieldManager( fm_ivar.nbfields() ).get_id2index();
    }
};

} // namespace dyablo