#pragma once

#include "kokkos_shared.h"

namespace dyablo {

class ForeachParticle;

class ParticleArray
{
    friend ForeachParticle;
    friend UserData;
public:
    using ParticleIndex = uint32_t;
    ParticleArray() = default;
    ParticleArray( const ParticleArray& pa ) = default;
    ParticleArray( ParticleArray&& pa ) = default;
    ParticleArray& operator=( const ParticleArray& pa ) = default;
    ParticleArray& operator=( ParticleArray&& pa ) = default;    

    ParticleArray( const std::string& name, uint32_t count )
    : particle_position( name, count, 3 )
    {}

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

protected:
    Kokkos::View< real_t**, Kokkos::LayoutLeft > particle_position;
};

class ParticleData : public ParticleArray
{
    friend ForeachParticle;
    friend UserData;
public:
    ParticleData() = default;
    ParticleData( const ParticleData& pa ) = default;
    ParticleData( ParticleData&& pa ) = default;
    ParticleData& operator=( const ParticleData& pa ) = default;
    ParticleData& operator=( ParticleData&& pa ) = default;  

    ParticleData( const ParticleArray& pa, const FieldManager& fieldManager )
    : ParticleArray( pa ),
      particle_data( this->particle_position.label()+"_data", pa.getNumParticles(), fieldManager.nbfields() ),
      fm( fieldManager.get_id2index() )
    {}

    ParticleData( const std::string& name, uint32_t nbParticles, const FieldManager& fieldManager )
    : ParticleArray( name, nbParticles ),
      particle_data( this->particle_position.label()+"_data", nbParticles, fieldManager.nbfields() ),
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
    Kokkos::View< real_t**, Kokkos::LayoutLeft > particle_data;
    id2index_t fm;
};

} // namespace dyablo