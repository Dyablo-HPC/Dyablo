#pragma once

#include "morton_utils.h"
#include "kokkos_shared.h"
#include "amr/AMRmesh.h"
#include "amr/LightOctree_storage.h"
#include "Kokkos_UnorderedMap.hpp"
#include "Kokkos_Macros.hpp"

#include "amr/LightOctree_base.h"
#include "amr/LightOctree_pablo.h"

namespace dyablo { 



class LightOctree_hashmap : public LightOctree_base, protected LightOctree_storage<>{
public:
    using LightOctree_base::OctantIndex;
    using LightOctree_base::pos_t;

    LightOctree_hashmap() = default;
    LightOctree_hashmap(const LightOctree_hashmap& lmesh) = default;

    template < typename AMRmesh_t >
    LightOctree_hashmap( const AMRmesh_t* pmesh, uint8_t level_min, uint8_t level_max )
    : LightOctree_storage( *pmesh ),
      oct_map(pmesh->getNumOctants()+pmesh->getNumGhosts()),
      min_level(level_min), max_level(level_max)
    {
        std::cout << "LightOctree rehash ..." << std::endl;
    
        const LightOctree_hashmap& this_ref = *this;
        const oct_map_t& oct_map = this->oct_map;
        uint32_t nbOcts = getNumOctants();
        uint32_t numOctants_tot = nbOcts + getNumGhosts();

        // Put octants into hashmap on device
        Kokkos::parallel_for( "LightOctree_hashmap::hash",
                              Kokkos::RangePolicy<>(0, numOctants_tot),
                              KOKKOS_LAMBDA(uint32_t ioct_local)
        {   
            OctantIndex iOct = OctantIndex::iOctLocal_to_OctantIndex( ioct_local, nbOcts );

            auto logical_pos = this_ref.get_logical_coords( iOct );
            level_t level = this_ref.getLevel( iOct );
            key_t logical_coords;
            logical_coords.level = level;
            logical_coords.i = logical_pos[IX];
            logical_coords.j = logical_pos[IY];
            logical_coords.k = logical_pos[IZ];           

            #ifndef NDEBUG
            oct_map_t::insert_result inserted = 
            #endif
            oct_map.insert( logical_coords, iOct );
            assert(inserted.success());
        });
    }

    using LightOctree_storage::getNumOctants;
    using LightOctree_storage::getNumGhosts;
    using LightOctree_storage::getNdim;
    using LightOctree_storage::getCenter;
    using LightOctree_storage::getCorner;
    using LightOctree_storage::getSize;
    using LightOctree_storage::getLevel;
    using LightOctree_storage::getBound;
    
    //! @copydoc LightOctree_base::findNeighbors()
    KOKKOS_INLINE_FUNCTION NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset )  const
    {
        //assert( !iOct.isGhost );

        if( this->isBoundary(iOct, offset) )
            return NeighborList{0,{}};

        // Get logical coordinates of neighbor
        
        level_t level = getLevel(iOct);
        auto lc = get_logical_coords(iOct);
        logical_coord_t octant_count = cell_count( level );
        key_t logical_coords;
        logical_coords.level = getLevel(iOct);
        logical_coords.i = (lc[IX] + octant_count + offset[IX]) % octant_count; // Periodic coord only works if offset > -octant_count
        logical_coords.j = (lc[IY] + octant_count + offset[IY]) % octant_count;
        logical_coords.k = (lc[IZ] + octant_count + offset[IZ]) % octant_count;   

        NeighborList res = {0};
        // Search octant at same level
        auto it = oct_map.find(logical_coords);
        if( oct_map.valid_at(it) )
        {
            // Found at same level
            res =  NeighborList{1, {oct_map.value_at(it)}};
        }
        else
        {
            key_t logical_coords_bigger;
            logical_coords_bigger.level = logical_coords.level-1;
            logical_coords_bigger.i = logical_coords.i >> 1;
            logical_coords_bigger.j = logical_coords.j >> 1;
            logical_coords_bigger.k = logical_coords.k >> 1;

            // Search octant at coarser level
            auto it = oct_map.find(logical_coords_bigger);
            if( oct_map.valid_at(it) ) 
            {
                // Found at coarser level
                res = NeighborList{1, {oct_map.value_at(it)}};
            }
            else
            {
                // Neighbor(s) is(are) at finer level
                assert(level+1 <= max_level);
                
                // Compute logical coord of first neighbor
                key_t logical_coords_smaller_origin;
                logical_coords_smaller_origin.level = logical_coords.level+1;
                logical_coords_smaller_origin.i = (logical_coords.i << 1) + (offset[IX]==-1);
                logical_coords_smaller_origin.j = (logical_coords.j << 1) + (offset[IY]==-1);
                logical_coords_smaller_origin.k = (logical_coords.k << 1) + (offset[IZ]==-1);
                int sz_max = (ndim==2) ? 0 : (offset[IZ]==0); // No offset in z in 2D
                int sy_max = (offset[IY]==0);
                int sx_max = (offset[IX]==0); // Constrained to plane adjacent to neighbor if offset in this direction
                
                for( int sz=0; sz<=sz_max; sz++ )
                for( int sy=0; sy<=sy_max; sy++ )
                for( int sx=0; sx<=sx_max; sx++ )
                {
                    res.m_size++;
                    key_t logical_coords_smaller = logical_coords_smaller_origin;
                    logical_coords_smaller.i += sx;
                    logical_coords_smaller.j += sy;
                    logical_coords_smaller.k += sz;
                    auto it = oct_map.find(logical_coords_smaller);
                    assert(oct_map.valid_at(it)); // Could not find neighbor
                    res.m_neighbors[res.m_size-1] = oct_map.value_at(it);
                }
                assert(res.m_size<=2*(ndim-1));
            }            
        }
        return res;
    }
    /// @copydoc LightOctree_base::isBoundary()
    KOKKOS_INLINE_FUNCTION
    bool isBoundary(const OctantIndex& iOct, const offset_t& offset) const {
      //assert( !iOct.isGhost );
      real_t dh = this->getSize(iOct);
      pos_t center = this->getCenter(iOct);    
      pos_t pos {
          center[IX] + offset[IX]*dh,
          center[IY] + offset[IY]*dh,
          center[IZ] + offset[IZ]*dh
      };
  
      //       Not periodic   and     not inside domain
      // in at least one dimension
      return (!this->is_periodic[IX] && !( 0<pos[IX] && pos[IX]<1 ))
          || (!this->is_periodic[IY] && !( 0<pos[IY] && pos[IY]<1 ))
          || (!this->is_periodic[IZ] && !( 0<=pos[IZ] && pos[IZ]<1 )) ;            
    }

    // ------------------------
    // Only in LightOctree_hashmap
    // ------------------------
    /**
     * Get octant from logical position
     **/
    KOKKOS_INLINE_FUNCTION
    OctantIndex getiOctFromCoordinates(uint16_t ix, uint16_t iy, uint16_t iz, uint16_t level) const
    {
        assert( ix < cell_count( level )  );
        assert( iy < cell_count( level )  );
        if(ndim == 3)
            assert( iz < cell_count( level )  );
        else 
            assert( iz == 0  );

        auto it = oct_map.find({level, ix, iy, iz});

        assert( oct_map.valid_at(it) );

        return oct_map.value_at(it);
    }
    /**
     * Get octant containing position pos
     **/
    KOKKOS_INLINE_FUNCTION
    OctantIndex getiOctFromPos(const pos_t& pos) const
    {
        assert( 0 < pos[IX] && pos[IX] < 1 );
        assert( 0 < pos[IY] && pos[IY] < 1 );
        if(ndim == 3)
            assert( 0 < pos[IZ] && pos[IZ] < 1 );
        else
            assert( pos[IZ] == 0 );

        key_t logical_coords;
        {
            uint32_t octant_count = cell_count( max_level );
            real_t octant_size = 1.0/octant_count;
            logical_coords.level = max_level;
            logical_coords.i = std::floor(pos[IX]/octant_size);
            logical_coords.j = std::floor(pos[IY]/octant_size);
            logical_coords.k = (ndim-2)*std::floor(pos[IZ]/octant_size);
        }

        for(level_t level=max_level; level>=min_level; level--)
        {
            auto it = oct_map.find(logical_coords);

            if( oct_map.valid_at(it) ) 
            {
                return oct_map.value_at(it);
            }
            logical_coords.level = logical_coords.level-1;
            logical_coords.i = logical_coords.i >> 1;
            logical_coords.j = logical_coords.j >> 1;
            logical_coords.k = logical_coords.k >> 1;
        }

        assert(false); //Could not find octant at this position
        return {};
    }

    using logical_coord_t = uint32_t;
    using level_t = logical_coord_t;
    struct key_t //! key type for hashmap (morton+level)
    {
        logical_coord_t level, i, j, k;
    };
private:
    using oct_ref_t = OctantIndex; //! value type for the hashmap
    using oct_map_t = Kokkos::UnorderedMap<key_t, oct_ref_t>; //! hashmap returning an octant form a key
    oct_map_t oct_map; //! hashmap returning an octant form a key

    level_t min_level; //! Coarser level of the octree
    level_t max_level; //! Finer level of the octree
};

} //namespace dyablo
