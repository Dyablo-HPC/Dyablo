#pragma once

#include "shared/morton_utils.h"
#include "shared/kokkos_shared.h"
#include "shared/amr/AMRmesh.h"
#include "Kokkos_UnorderedMap.hpp"
#include "Kokkos_Macros.hpp"

#include "shared/amr/LightOctree_base.h"
#include "shared/amr/LightOctree_pablo.h"

namespace dyablo { 

class LightOctree_hashmap : public LightOctree_base{
public:
    LightOctree_hashmap() = default;
    LightOctree_hashmap(const LightOctree_hashmap& lmesh) = default;

    template < typename AMRmesh_t >
    LightOctree_hashmap( const AMRmesh_t* pmesh, uint8_t level_min, uint8_t level_max )
    : oct_map(pmesh->getNumOctants()+pmesh->getNumGhosts()),
      oct_data("LightOctree::oct_data", pmesh->getNumOctants()+pmesh->getNumGhosts(), OCT_DATA_COUNT),
      numOctants(pmesh->getNumOctants()) , min_level(level_min), max_level(level_max), ndim(pmesh->getDim())
    {
        is_periodic[IX] = pmesh->getPeriodic(IX);
        is_periodic[IY] = pmesh->getPeriodic(IY);
        is_periodic[IZ] = pmesh->getPeriodic(IZ);
        std::cout << "LightOctree rehash ..." << std::endl;
        init(pmesh, oct_data, oct_map, numOctants, min_level, max_level);
    }
    //! @copydoc LightOctree_base::getNumOctants()
    KOKKOS_INLINE_FUNCTION uint32_t getNumOctants() const
    {
        return numOctants;
    }
    //! @copydoc LightOctree_base::getNdim()
    KOKKOS_INLINE_FUNCTION uint8_t getNdim() const
    {
        return ndim;
    }
    //! @copydoc LightOctree_base::getCenter()
    KOKKOS_INLINE_FUNCTION pos_t getCenter(const OctantIndex& iOct)  const
    {
        pos_t pos = getCorner(iOct);
        real_t oct_size = getSize(iOct);
        return {
            pos[IX] + oct_size/2,
            pos[IY] + oct_size/2,
            pos[IZ] + (ndim-2)*(oct_size/2)
        };
    }
    //! @copydoc LightOctree_base::getCorner()
    KOKKOS_INLINE_FUNCTION pos_t getCorner(const OctantIndex& iOct)  const
    {
        return {
            oct_data(get_ioct_local(iOct), ICORNERX),
            oct_data(get_ioct_local(iOct), ICORNERY),
            oct_data(get_ioct_local(iOct), ICORNERZ),
        };
    }
    KOKKOS_INLINE_FUNCTION bool getBound(const OctantIndex& iOct)  const
    {
         return oct_data(get_ioct_local(iOct), ISBOUND);
    }
    //! @copydoc LightOctree_base::getSize()
    KOKKOS_INLINE_FUNCTION real_t getSize(const OctantIndex& iOct)  const
    {
        return 1.0/std::pow( 2, getLevel(iOct) );
    }
    //! @copydoc LightOctree_base::getLevel()
    KOKKOS_INLINE_FUNCTION uint8_t getLevel(const OctantIndex& iOct)  const
    {
        return oct_data(get_ioct_local(iOct), ILEVEL);
    }
    //! @copydoc LightOctree_base::findNeighbors()
    KOKKOS_INLINE_FUNCTION NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset )  const
    {
        assert( !iOct.isGhost );

        if( this->isBoundary(iOct, offset) )
            return NeighborList{0,{}};

        // Compute physical position of neighbor
        pos_t c = getCenter(iOct);
        uint8_t level = getLevel(iOct); 
        uint32_t octant_count = std::pow( 2, level );
        real_t octant_size = 1.0/octant_count;
        real_t eps = octant_size/8;
        // Compute logical octant position at this level
        auto periodic_coord = KOKKOS_LAMBDA(real_t pos, int8_t offset) -> int32_t
        {
            int32_t grid_pos = std::floor((pos+eps)/octant_size) + offset;
            return (grid_pos+octant_count) % octant_count; // Only works if grid_pos>-octant_count
        };
        index_t<3> logical_coords = {
            periodic_coord( c[IX], offset[IX] ),
            periodic_coord( c[IY], offset[IY] ),
            periodic_coord( c[IZ], offset[IZ] )
        };

        // Get morton at same level
        morton_t morton_neighbor = compute_morton_key( logical_coords );

        NeighborList res = {0};
        // Search octant at same level
        auto it = oct_map.find(get_key(level, morton_neighbor));
        if( oct_map.valid_at(it) )
        {
            // Found at same level
            res =  NeighborList{1, {oct_map.value_at(it)}};
        }
        else
        {
            morton_t morton_neighbor_bigger = morton_neighbor >> 3; // Compute morton at coarser level : remove last 3 bits
            // Search octant at coarser level
            auto it = oct_map.find(get_key(level-1, morton_neighbor_bigger));
            if( oct_map.valid_at(it) ) 
            {
                // Found at coarser level
                res = NeighborList{1, {oct_map.value_at(it)}};
            }
            else
            {
                // Neighbor(s) is(are) at finer level
                assert(level+1 <= max_level);

                for( uint8_t x=0; x<2; x++ )
                for( uint8_t y=0; y<2; y++ )
                for( uint8_t z=0; z<(ndim-1); z++ )
                {
                    // The number of smaller neighbors is hard to determine (ex : only one smaller neighbor in corners)
                    // Add smaller neighbor only if near original octant
                    // direction is unsconstrained OR offset left + suboctant right OR offset right + suboctant left
                    // (offset[IX] == 0)           OR (offset[IX]==-1 && x==1)      OR (offset[IX]==1 && x==0)
                    
                    if( ( (offset[IX] == 0) or (offset[IX]==-1 && x==1) or (offset[IX]==1 && x==0) )
                    and ( (offset[IY] == 0) or (offset[IY]==-1 && y==1) or (offset[IY]==1 && y==0) )
                    and ( (offset[IZ] == 0) or (offset[IZ]==-1 && z==1) or (offset[IZ]==1 && z==0) ) )
                    {
                        res.m_size++;
                        assert(res.m_size<=4);
                        // Morton at level+1 is morton at level with 3 more bits (8 suboctants)
                        morton_t morton_neighbor_smaller = ( morton_neighbor << 3 ) + (z << IZ) + (y << IY) + (x << IX);
                        // Get the smaller neighbor (which necessarily exist)
                        auto it = oct_map.find(get_key(level+1, morton_neighbor_smaller));
                        assert(oct_map.valid_at(it)); // Could not find neighbor
                        res.m_neighbors[res.m_size-1] = oct_map.value_at(it);
                    }
                }
            }            
        }
        return res;
    }
    /// @copydoc LightOctree_base::isBoundary()
    KOKKOS_INLINE_FUNCTION
    bool isBoundary(const OctantIndex& iOct, const offset_t& offset) const {
      assert( !iOct.isGhost );
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
        assert( ix < std::pow( 2, level )  );
        assert( iy < std::pow( 2, level )  );
        if(ndim == 3)
            assert( iz < std::pow( 2, level )  );
        else 
            assert( iz == 0  );

        morton_t morton = compute_morton_key( ix, iy, iz );
        auto it = oct_map.find(get_key(level, morton));

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
        morton_t morton;
        {
            index_t<3> logical_coords;
            uint32_t octant_count = std::pow( 2, max_level );
            real_t octant_size = 1.0/octant_count;
            logical_coords[IX] = std::floor(pos[IX]/octant_size);
            logical_coords[IY] = std::floor(pos[IY]/octant_size);
            logical_coords[IZ] = (ndim-2)*std::floor(pos[IZ]/octant_size);

            morton = compute_morton_key( logical_coords );
        }

        for(uint8_t level=max_level; level>=min_level; level--)
        {
            auto it = oct_map.find(get_key(level, morton));

            if( oct_map.valid_at(it) ) 
            {
                return oct_map.value_at(it);
            }

            morton = morton >> 3;
        }

        assert(false); //Could not find octant at this position
        return {};
    }

private:
    using morton_t = uint64_t; //! Type of morton index
    using level_t = uint8_t; //! Type for level
    using key_t = uint64_t; //! key type for hashmap (morton+level)
    using oct_ref_t = OctantIndex; //! value type for the hashmap
    using oct_map_t = Kokkos::UnorderedMap<key_t, oct_ref_t>; //! hashmap returning an octant form a key
    oct_map_t oct_map; //! hashmap returning an octant form a key

    //! Index to access different fields in `oct_data`
    enum oct_data_field_t{
        ICORNERX, 
        ICORNERY, 
        ICORNERZ, 
        ILEVEL,
        ISBOUND,
        OCT_DATA_COUNT
    };
    using oct_data_t = DataArray;
    //! Kokkos::view containing octants position and level 
    //! ex: (oct_data(iOct, ILEVEL) is octant level)
    oct_data_t oct_data;    

    //! Construct a hashmap key from a level and a morton index
    KOKKOS_INLINE_FUNCTION static key_t get_key( level_t level, morton_t morton ) {
        constexpr uint8_t shift = sizeof(level)*8;
        key_t res = (morton << shift) + level; 
        assert( morton == (res >> shift) ); // Loss of data from shift
        return res;
    };
    
    //! Get octant index in oct_data from an OctantIndex
    KOKKOS_INLINE_FUNCTION uint32_t get_ioct_local(const OctantIndex& oct) const
    {
        // Ghosts are stored after non-ghosts
        return OctantIndex::OctantIndex_to_iOctLocal(oct, numOctants);
    }

    uint32_t numOctants; //! Number of local octants (no ghosts)
    level_t min_level; //! Coarser level of the octree
    level_t max_level; //! Finer level of the octree
    int ndim; //! 2D or 3D 
    Kokkos::Array<bool,3> is_periodic;   
    
public: // init() has to be public for KOKKOS_LAMBDA

    /**
     * Fetches data from pmesh and fill hashmap
     **/
    // TODO make a specialization for AMRmesh_hashmap that directly copies Kokkos::Arrays
    template < typename AMRmesh_t >
    static void init(const AMRmesh_t* pmesh, oct_data_t& oct_data, oct_map_t& oct_map, uint32_t numOctants, uint8_t min_level, uint8_t max_level)
    {   
        const uint32_t numOctants_tot = pmesh->getNumOctants()+pmesh->getNumGhosts();
        
        // Copy mesh data from PABLO tree to LightOctree
        {   
            oct_data_t::HostMirror oct_data_host = Kokkos::create_mirror_view(oct_data);

            Kokkos::parallel_for( "LightOctree_hashmap::copydata", 
                                Kokkos::RangePolicy<Kokkos::OpenMP>(0, numOctants_tot),
                                [=]( uint32_t ioct_local )
            {
                OctantIndex oct = OctantIndex::iOctLocal_to_OctantIndex( ioct_local, numOctants );

                auto c = oct.isGhost ? 
                        pmesh->getCoordinatesGhost(oct.iOct):
                        pmesh->getCoordinates(oct.iOct);
                uint8_t level = oct.isGhost ? 
                        pmesh->getLevelGhost(oct.iOct):
                        pmesh->getLevel(oct.iOct);
                bool is_bound = !oct.isGhost && pmesh->getBound(oct.iOct);

                oct_data_host(ioct_local, ICORNERX) = c[IX];
                oct_data_host(ioct_local, ICORNERY) = c[IY];
                oct_data_host(ioct_local, ICORNERZ) = c[IZ];
                oct_data_host(ioct_local, ILEVEL) = level;
                oct_data_host(ioct_local, ISBOUND) = is_bound;
            });

            // Copy data to device
            Kokkos::deep_copy(oct_data,oct_data_host);
        }

        // Put octants into hashmap on device
        Kokkos::parallel_for( "LightOctree_hashmap::hash",
                              Kokkos::RangePolicy<>(0, numOctants_tot),
                              KOKKOS_LAMBDA(uint32_t ioct_local)
        {
            pos_t c;
            c[IX] = oct_data(ioct_local, static_cast<int>(ICORNERX));
            c[IY] = oct_data(ioct_local, ICORNERY);
            c[IZ] = oct_data(ioct_local, ICORNERZ);
            uint8_t level = oct_data(ioct_local, ILEVEL);

            uint32_t octant_count = std::pow( 2, level );
            real_t octant_size = 1.0/octant_count;
            real_t eps = octant_size/8; // To avoid rounding error when computing logical coords
                auto periodic_coord = [=](real_t pos) -> int32_t
            {
                int32_t grid_pos = std::floor((pos+eps)/octant_size);
                return (grid_pos+octant_count) % octant_count; // Only works if grid_pos>-octant_count
            };
            index_t<3> logical_coords = {
                periodic_coord( c[IX] ),
                periodic_coord( c[IY] ),
                periodic_coord( c[IZ] )
            };
            morton_t morton = compute_morton_key(logical_coords);

            OctantIndex ioct = {ioct_local, false};
            if( ioct_local >= numOctants )
            {
                ioct.iOct -= numOctants;
                ioct.isGhost = true;
            }

            oct_map_t::insert_result inserted = oct_map.insert( get_key(level, morton), ioct );
            assert(inserted.success());
        });
    }
};

} //namespace dyablo
