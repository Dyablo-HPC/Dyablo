#pragma once

#include "kokkos_shared.h"

namespace dyablo {

class ForeachParticle;

class ParticleArray
{
    friend ForeachParticle;
public:
    using ParticleIndex = uint32_t;

    ParticleArray( const std::string& name, uint32_t count, const FieldManager& fieldManager )
    : particle_position( name+"_pos", count, 3 ),
      particle_data( name+"_data", count, fieldManager.nbfields() ),
      fm( fieldManager.get_id2index() )
    {}

    FieldManager field_manager()
    {
        return FieldManager(fm.enabled_fields());
    }

    KOKKOS_INLINE_FUNCTION
    int nbfields() const
    {
        return fm.nbfields();
    }

    KOKKOS_INLINE_FUNCTION
    uint32_t getNumParticles() const 
    {
        return particle_position.extent(0);
    }

    KOKKOS_INLINE_FUNCTION
    real_t& pos( const ParticleIndex& iPart, ComponentIndex3D iDir ) const
    {
        return particle_position(iPart, iDir);
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at( const ParticleIndex& iPart, VarIndex field ) const
    {
        return at_ivar( iPart, fm[field] );
    }

    KOKKOS_INLINE_FUNCTION
    real_t& at_ivar( const ParticleIndex& iPart, int ivar ) const
    {
        return particle_data( iPart, ivar );
    }
private:
    Kokkos::View< real_t**, Kokkos::LayoutLeft > particle_position;
    Kokkos::View< real_t**, Kokkos::LayoutLeft > particle_data;
    id2index_t fm;
};

} // namespace dyablo