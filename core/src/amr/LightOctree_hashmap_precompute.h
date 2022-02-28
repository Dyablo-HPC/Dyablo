#pragma once

#include "amr/LightOctree_hashmap.h"

namespace dyablo { 

namespace{
KOKKOS_INLINE_FUNCTION 
static uint32_t offset_to_index(const LightOctree_base::offset_t& offset, uint32_t ndims) 
{
    return offset[IX]+1 + 3*(offset[IY]+1) + (ndims==3)*3*3*(offset[IZ]+1);
}

KOKKOS_INLINE_FUNCTION 
static LightOctree_base::offset_t index_to_offset(uint32_t index, uint32_t ndims)
{
    int8_t iz = index / (3*3);
    int8_t iy = (index - 3*3*iz)/3;
    int8_t ix = index % 3;

    return LightOctree_base::offset_t{ (int8_t)(ix-1), (int8_t)(iy-1), (int8_t)((ndims==2)?0:iz-1)};
}

void LightOctree_hashmap_precompute_init( const LightOctree_hashmap& lmesh_hashmap, 
    Kokkos::View< uint8_t**, Kokkos::LayoutLeft >& neighbors_count, 
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbors_offset,
    Kokkos::View< uint32_t* >& neighbors_iOct, 
    uint32_t ndims )
{
    using NeighborList = LightOctree_hashmap::NeighborList;

    uint32_t nneighbors = (ndims==2) ? 3*3 : 3*3*3;
    uint32_t nbOcts = lmesh_hashmap.getNumOctants();

    // Compute number of neighbors at each offset for each octant
    neighbors_count = Kokkos::View< uint8_t**, Kokkos::LayoutLeft >("LightOctree_hashmap_precompute::neighbors_count", nbOcts, nneighbors);
    uint32_t nneigh_tot = 0;
    Kokkos::parallel_reduce( "LightOctree_hashmap_precompute::precompute_count", neighbors_count.size(),
                          KOKKOS_LAMBDA( uint32_t index, uint32_t& nneigh_partial )
    {
        uint32_t iOct = index%nbOcts;
        uint32_t offset_index = index/nbOcts;
        LightOctree_base::offset_t offset = index_to_offset(offset_index, ndims);
        if(offset[IX]!=0 || offset[IY]!=0 || offset[IZ]!=0 )
        {
            NeighborList ns = lmesh_hashmap.findNeighbors({iOct,false}, offset);
            neighbors_count(iOct, offset_index) = ns.size();
            nneigh_partial += ns.size();
        }
    } , nneigh_tot );

    neighbors_offset = Kokkos::View< uint32_t**, Kokkos::LayoutLeft >("LightOctree_hashmap_precompute::neighbors_offset", nbOcts, nneighbors);
    neighbors_iOct = Kokkos::View< uint32_t* >("LightOctree_hashmap_precompute::neighbors_iOct", nneigh_tot);
    Kokkos::parallel_scan( "LightOctree_hashmap_precompute::precompute neighborhood", neighbors_count.size(),
                              KOKKOS_LAMBDA(uint32_t index, uint32_t& iOct_offset, const bool final)
    {
        uint32_t iOct = index/nneighbors;
        uint32_t offset_index = index%nneighbors;
        LightOctree_base::offset_t offset = index_to_offset(offset_index, ndims);

        if(offset[IX]!=0 || offset[IY]!=0 || offset[IZ]!=0 )
        {
            uint8_t nneighbors = neighbors_count(iOct, offset_index);          
            if(final)
            {
                NeighborList ns = lmesh_hashmap.findNeighbors({iOct,false}, offset);
                neighbors_offset(iOct, offset_index) = iOct_offset;
                for(int i=0; i<nneighbors; i++)
                {
                    neighbors_iOct(iOct_offset+i) = LightOctree_base::OctantIndex::OctantIndex_to_iOctLocal(ns[i], nbOcts);
                }                
            }
            iOct_offset += nneighbors;
        }
    });
}
} // namespace

class LightOctree_hashmap_precompute : public LightOctree_hashmap{
public:
    LightOctree_hashmap_precompute() = default;
    LightOctree_hashmap_precompute(const LightOctree_hashmap_precompute& lmesh) = default;

    template < typename AMRmesh_t >
    LightOctree_hashmap_precompute( const AMRmesh_t* pmesh, uint8_t level_min, uint8_t level_max )
    : LightOctree_hashmap(pmesh, level_min, level_max)
    {
        LightOctree_hashmap_precompute_init(*this, neighbors_count, neighbors_offset, neighbors_iOct, getNdim());
    }
    
    //! @copydoc LightOctree_base::findNeighbors()
    KOKKOS_INLINE_FUNCTION NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset )  const
    {
        assert( !iOct.isGhost );

        int ndim = getNdim();

        uint8_t nneighbors = neighbors_count(iOct.iOct, offset_to_index(offset, ndim));
        uint32_t n_offset = neighbors_offset(iOct.iOct, offset_to_index(offset, ndim));
        Kokkos::Array<OctantIndex,4> neighbors;        
        for( int i=0; i<nneighbors; i++ )
        {
            neighbors[i] = LightOctree_base::OctantIndex::iOctLocal_to_OctantIndex( neighbors_iOct( n_offset + i ), getNumOctants() );
        }
        return NeighborList{ nneighbors, neighbors };
    }

private:
    /// Number of neighbors for (iOct, offset_index)
    Kokkos::View< uint8_t**, Kokkos::LayoutLeft > neighbors_count;
    /// First neighbor in neighbors_iOct for (iOct, offset_index)
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbors_offset;
    /// Neighbors for n=(iOct, offset_index) are 
    /// [ neighbors_iOct( neighbors_offset(n) ), ..., neighbors_iOct( neighbors_offset(n) + neighbors_count(n) ) ]
    Kokkos::View< uint32_t* > neighbors_iOct; 
};

} //namespace dyablo
