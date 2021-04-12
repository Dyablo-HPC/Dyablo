#pragma once

#include "shared/amr/LightOctree_hashmap.h"

namespace dyablo { 

namespace{
KOKKOS_INLINE_FUNCTION 
static uint32_t offset_to_index(const LightOctree_base::offset_t& offset) 
{
    return offset[IX]+1 + 3*(offset[IY]+1) + 3*3*(offset[IZ]+1);
}

KOKKOS_INLINE_FUNCTION 
static LightOctree_base::offset_t index_to_offset(uint32_t index)
{
    int8_t iz = index / (3*3);
    int8_t iy = (index - 3*3*iz)/3;
    int8_t ix = index % 3;

    return LightOctree_base::offset_t{ ix-1, iy-1, iz-1};
}

void LightOctree_hashmap_precompute_init( const LightOctree_hashmap& lmesh_hashmap, Kokkos::View< LightOctree_base::NeighborList** >& neighbors_precompute )
{
    Kokkos::parallel_for( "LightOctree_hashmap_precompute::precompute neighborhood", neighbors_precompute.size(),
                              KOKKOS_LAMBDA(uint32_t index)
    {
        uint32_t iOct = index/(3*3*3);
        uint32_t offset_index = index%(3*3*3);
        LightOctree_base::offset_t offset = index_to_offset(offset_index);

        if(offset[IX]!=0 || offset[IY]!=0 || offset[IZ]!=0 )
        {
            neighbors_precompute(iOct, offset_index) = lmesh_hashmap.findNeighbors({iOct,false}, offset);
        }
    });
}
} // namespace

class LightOctree_hashmap_precompute : public LightOctree_hashmap{
public:
    LightOctree_hashmap_precompute() = default;
    LightOctree_hashmap_precompute(const LightOctree_hashmap_precompute& lmesh) = default;

    template < typename AMRmesh_t >
    LightOctree_hashmap_precompute( std::shared_ptr<AMRmesh_t> pmesh, uint8_t level_min, uint8_t level_max )
    : LightOctree_hashmap(pmesh, level_min, level_max), neighbors_precompute("neighbors_precompute", pmesh->getNumOctants(), 3*3*3)
    {
        LightOctree_hashmap_precompute_init(*this, this->neighbors_precompute);
    }
    
    //! @copydoc LightOctree_base::findNeighbors()
    KOKKOS_INLINE_FUNCTION NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset )  const
    {
        assert( !iOct.isGhost );

        return neighbors_precompute(iOct.iOct, offset_to_index(offset));
    }

private:
    Kokkos::View< NeighborList** > neighbors_precompute; 
};

} //namespace dyablo
