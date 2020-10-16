#pragma once

#include <memory>
#include <unordered_map>

#include "shared/morton_utils.h"
#include "shared/kokkos_shared.h"
#include "shared/bitpit_common.h"
#include "shared/HydroParams.h"
#include "Kokkos_UnorderedMap.hpp"

namespace dyablo { 
namespace muscl_block {

/**
 * Table to convert neighbor relative position to "iface" parameter for findNeighbors()
 * If neighbor octant is at position (x, y, z) relative to local octant (-1<=x,y,z<=1)
 * iface for findNeighbors() is iface_from_pos[z+1][y+1][x+1]-1
 * 
 * In 2D : offset = (x,y,0) => iface = iface_from_pos[0][y+1][x+1]-1
 * 
 * Has been generated by https://gitlab.maisondelasimulation.fr/pkestene/dyablo/-/snippets/3
 **/
static constexpr uint8_t iface_from_pos[3][3][3] = {
                        {{ 1, 3, 2 },
                            { 1, 5, 2 },
                            { 3, 4, 4 }},
                        {{ 5, 3, 6 },
                            { 1, 0, 2 },
                            { 7, 4, 8 }},
                        {{ 5, 11, 6 },
                            { 9, 6, 10 },
                            { 7, 12, 8 }}};

class LightOctree_base{
public:
    struct OctantIndex
    {
        uint32_t iOct;
        bool isGhost;
    };
    using pos_t = Kokkos::Array<real_t,3>;
    using offset_t = Kokkos::Array<int8_t,3>;
    struct NeighborList
    {
        uint8_t m_size;
        Kokkos::Array<OctantIndex,4> m_neighbors;

        KOKKOS_INLINE_FUNCTION uint8_t size() const
        {
            return m_size;
        }
        KOKKOS_INLINE_FUNCTION const OctantIndex& operator[](uint8_t i) const
        {
            return m_neighbors[i];
        }
    };
};

class LightOctree_pablo : public LightOctree_base{
public:
    LightOctree_pablo( std::shared_ptr<AMRmesh> pmesh, const HydroParams& params )
    : pmesh(pmesh), ndim(pmesh->getDim())
    {}
    uint32_t getNumOctants() const
    {
        return pmesh->getNumOctants();
    }
    bool getBound(const OctantIndex& iOct)  const
    {
        assert( !iOct.isGhost );
        return pmesh->getBound(iOct.iOct);
    }
    pos_t getCenter(const OctantIndex& iOct)  const
    {
        bitpit::darray3 pcenter = iOct.isGhost ? pmesh->getCenter(pmesh->getGhostOctant(iOct.iOct)) : pmesh->getCenter(iOct.iOct);
        return {pcenter[0], pcenter[1], pcenter[2]};
    }
    pos_t getCorner(const OctantIndex& iOct)  const
    {
        bitpit::darray3 pmin = iOct.isGhost ? 
                pmesh->getCoordinates(pmesh->getGhostOctant(iOct.iOct)) : 
                pmesh->getCoordinates(iOct.iOct);
        return {pmin[IX], pmin[1], pmin[2]};
    }
    real_t getSize(const OctantIndex& iOct)  const
    {
        real_t oct_size = iOct.isGhost ? 
                pmesh->getSize(pmesh->getGhostOctant(iOct.iOct)) : 
                pmesh->getSize(iOct.iOct);
        return oct_size;
    }
    uint8_t getLevel(const OctantIndex& iOct)  const
    {
        uint8_t oct_level = iOct.isGhost ? 
                pmesh->getLevel(pmesh->getGhostOctant(iOct.iOct)) : 
                pmesh->getLevel(iOct.iOct);
        return oct_level;
    }
    NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset )  const
    {
        assert( !iOct.isGhost );

        // Determine codimension
        int count_dims = 0;
        count_dims += std::abs(offset[IX]);
        count_dims += std::abs(offset[IY]);
        count_dims += std::abs(offset[IZ]);    
        uint8_t pablo_codim = count_dims;
        int neighbor_pos_z = (ndim==2) ? 0 : offset[IZ]+1;
        uint8_t pablo_iface = iface_from_pos[neighbor_pos_z][offset[IY]+1][offset[IX]+1] - 1;

        std::vector<uint32_t> iOct_neighbors;
        std::vector<bool> isghost_neighbors;

        pmesh->findNeighbours(iOct.iOct, pablo_iface, pablo_codim, iOct_neighbors, isghost_neighbors);

        NeighborList neighbors={0,{99,true}};
        if( iOct_neighbors.size() == 0 )
        {
            pos_t cellPos = getCenter(iOct);
            uint8_t level = getLevel(iOct);
            real_t cellSize = 1.0/std::pow(2, level );
            cellPos[IX] += offset[IX]*cellSize*0.6;
            cellPos[IY] += offset[IY]*cellSize*0.6;
            cellPos[IZ] += offset[IZ]*cellSize*0.6;
            bitpit::bvector periodic = pmesh->getPeriodic();
            if( ( periodic[2*IX] or ( 0.0 <= cellPos[IX] && cellPos[IX] < 1.0 ) )
             or ( periodic[2*IY] or ( 0.0 <= cellPos[IY] && cellPos[IY] < 1.0 ) )
             or ( periodic[2*IZ] or ( 0.0 <= cellPos[IZ] && cellPos[IZ] < 1.0 ) ) )
            {
                //Get periodic position inside domain
                if(periodic[2*IX]) cellPos[IX] -= std::floor(cellPos[IX]/1.0);
                if(periodic[2*IY]) cellPos[IY] -= std::floor(cellPos[IY]/1.0);
                if(periodic[2*IZ]) cellPos[IZ] -= std::floor(cellPos[IZ]/1.0);

                fix_missing_corner_neighbor(iOct.iOct, offset, cellPos, neighbors);
            }
        }
        else
        {
            neighbors.m_size = iOct_neighbors.size();
            for(int i=0; i<neighbors.m_size; i++)
            {
                neighbors.m_neighbors[i].iOct = iOct_neighbors[i];
                neighbors.m_neighbors[i].isGhost = isghost_neighbors[i];
            }
        }

        return neighbors;
    }

protected:
    std::shared_ptr<AMRmesh> pmesh;
    uint8_t ndim;

private:
    /** 
     * When neighbor is larger, it is returned in only one `findNeighbours()` request
     * Sometimes `findNeighbours()` edge/node returns 0 neighbors because the only neighbor has already been returned by findNeighbour
     * on a lower codimension request. In this case, actual edge/node neighbor has to be searched in lower codimention neighbor
     **/
    void fix_missing_corner_neighbor( uint32_t iOct_global, const offset_t& neighbor, const pos_t cellPos, NeighborList& res_neighbors) const
    {
        constexpr uint8_t CODIM_NODE = 3;
        constexpr uint8_t CODIM_EDGE = 2;
        //constexpr uint8_t CODIM_FACE = 1;

        /// Check if position `cellPos` is inside neighbor designated by search_codim and search_iface
        /// @returns true if cell is inside and sets res_iOct_neighbors and res_isGhost_neighbors accordingly, false otherwise
        auto check_neighbor = [&](const offset_t& offset)
        {   
             NeighborList neighbors = findNeighbors({iOct_global,false}, offset);
            if( neighbors.size() == 1 ) // Looking only for bigger neighbors
            {
                pos_t neigh_min = getCorner(neighbors[0]);
                real_t oct_size = getSize(neighbors[0]);

                pos_t neigh_max = {
                        neigh_min[IX] + oct_size,
                        neigh_min[IY] + oct_size,
                        neigh_min[IZ] + oct_size
                };

                if( neigh_min[IX] < cellPos[IX] && cellPos[IX] < neigh_max[IX] 
                &&  neigh_min[IY] < cellPos[IY] && cellPos[IY] < neigh_max[IY]
                && ((ndim==2) || (neigh_min[IZ] < cellPos[IZ] && cellPos[IZ] < neigh_max[IZ])) )
                {
                    res_neighbors = neighbors;
                    return true;
                }
            }
            return false;        
        };

        // Determine neighbor codimension (node/edge/face)
        int codim = 0;
        codim += std::abs(neighbor[IX]);
        codim += std::abs(neighbor[IY]);
        codim += std::abs(neighbor[IZ]);

        if(ndim == 2)
        {
            assert( codim == CODIM_EDGE ); // In 2D, only edges can have this issue
            // Search both faces connected to edge:
            if( check_neighbor({neighbor[IX],0           ,0}) ) return;
            if( check_neighbor({0,           neighbor[IY],0}) ) return;
            assert(false); // Failed to find neighbor...
        }
        else 
        {
            assert( codim == CODIM_NODE || codim == CODIM_EDGE ); // In 3D, edges and node can have this issue

            if(codim == CODIM_NODE) // search direction is a node (corner)
            {
                // Search the 3 edges connected to the node
                for(int i=0; i<3; i++)
                {
                    offset_t search_neighbor = {neighbor[IX], neighbor[IY], neighbor[IZ]};
                    search_neighbor[i] = 0; // Search edges that have 2 coordinate in common
                    if( check_neighbor(search_neighbor) ) return;
                }
            }
            // Search connected faces
            for(int i=0; i<3; i++)
            {
                offset_t search_neighbor = {0,0,0};
                search_neighbor[i] = neighbor[i]; // Search faces that have 1 coordinate in common
                if( search_neighbor[IX] != 0 || search_neighbor[IY] != 0 || search_neighbor[IZ] != 0 ) // Only 2 faces when initially edge (0,0,0 not a neighbor)
                {
                    if( check_neighbor(search_neighbor) ) return;
                }
            }
            assert(false); // Failed to find neighbor...
        }     
    } 
};

class LightOctree_hashmap : public LightOctree_base{
public:
    LightOctree_hashmap( std::shared_ptr<AMRmesh> pmesh, const HydroParams& params )
    : oct_map(pmesh->getNumOctants()+pmesh->getNumGhosts()),
      oct_data("LightOctree::oct_data", pmesh->getNumOctants()+pmesh->getNumGhosts(), OCT_DATA_COUNT),
      numOctants(pmesh->getNumOctants()) , max_level(params.level_max), ndim(pmesh->getDim())
    {
        init(pmesh, params);
    }

    uint32_t getNumOctants() const
    {
        return numOctants;
    }
    // bool getBound(const OctantIndex& iOct)  const
    // {
    //     assert( !iOct.isGhost );
    //     return pmesh->getBound(iOct.iOct);
    // }
    KOKKOS_INLINE_FUNCTION pos_t getCenter(const OctantIndex& iOct)  const
    {
        pos_t pos = getCorner(iOct);
        real_t oct_size = getSize(iOct);
        return {
            pos[IX] + oct_size,
            pos[IY] + oct_size,
            pos[IZ] + oct_size
        };
    }
    KOKKOS_INLINE_FUNCTION pos_t getCorner(const OctantIndex& iOct)  const
    {
        return {
            oct_data(get_ioct_local(iOct), ICORNERX),
            oct_data(get_ioct_local(iOct), ICORNERY),
            oct_data(get_ioct_local(iOct), ICORNERZ),
        };
    }
    KOKKOS_INLINE_FUNCTION real_t getSize(const OctantIndex& iOct)  const
    {
        return std::pow( 2, getLevel(iOct) );
    }
    KOKKOS_INLINE_FUNCTION uint8_t getLevel(const OctantIndex& iOct)  const
    {
        return oct_data(get_ioct_local(iOct), ILEVEL);
    }

    KOKKOS_INLINE_FUNCTION NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset )  const
    {
        assert( !iOct.isGhost );

        pos_t c = getCenter(iOct);
        uint8_t level = getLevel(iOct); 

        uint32_t octant_count = std::pow( 2, level );
        real_t octant_size = 1.0/octant_count;
        real_t eps = octant_size/8;

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
        auto it = oct_map.find(get_key(level, morton_neighbor));
        if( oct_map.valid_at(it) ) // Found at same level
        {
            // Found at same level
            res =  NeighborList{1, {oct_map.value_at(it)}};
        }
        else
        {
            morton_t morton_neighbor_bigger = morton_neighbor >> 3; // Remove last 3 bits

            auto it = oct_map.find(get_key(level-1, morton_neighbor_bigger));
            if( oct_map.valid_at(it) ) 
            {
                // Found at bigger level
                res = NeighborList{1, {oct_map.value_at(it)}};
            }
            else
            {
                assert(level+1 <= max_level);
                // Neighbor is smaller
                for( uint8_t x=0; x<2; x++ )
                for( uint8_t y=0; y<2; y++ )
                for( uint8_t z=0; z<(ndim-1); z++ )
                {
                    // Add smaller neighbor only if near original octant
                    // direction is unsconstrained OR offset left + suboctant right OR offset right + suboctant left
                    // (offset[IX] == 0)           OR (offset[IX]==-1 && x==1)      OR (offset[IX]==1 && x==0)
                    // offset\x 0 1
                    //   -1     F T
                    //    0     T T
                    //    1     T F
                    
                    if( ( (offset[IX] == 0) or (offset[IX]==-1 && x==1) or (offset[IX]==1 && x==0) )
                    and ( (offset[IY] == 0) or (offset[IY]==-1 && y==1) or (offset[IY]==1 && y==0) )
                    and ( (offset[IZ] == 0) or (offset[IZ]==-1 && z==1) or (offset[IZ]==1 && z==0) ) )
                    {
                        res.m_size++;
                        assert(res.m_size<=4);
                        morton_t morton_neighbor_smaller = ( morton_neighbor << 3 ) + (z << IZ) + (y << IY) + (x << IX);
                        auto it = oct_map.find(get_key(level+1, morton_neighbor_smaller));
                        assert(oct_map.valid_at(it)); // Could not find neighbor
                        res.m_neighbors[res.m_size-1] = oct_map.value_at(it);
                    }
                }
            }            
        }
        return res;
    }

private:
    using morton_t = uint64_t;
    using level_t = uint8_t;
    using key_t = uint64_t;
    using oct_ref_t = OctantIndex;
    using oct_map_t = Kokkos::UnorderedMap<key_t, oct_ref_t>;
    oct_map_t oct_map;
    int numOctants;

    enum oct_data_field_t{
        ICORNERX, 
        ICORNERY, 
        ICORNERZ, 
        ILEVEL,
        OCT_DATA_COUNT
    };

    using oct_data_t = DataArray;
    oct_data_t oct_data;
    

    KOKKOS_INLINE_FUNCTION static key_t get_key( level_t level, morton_t morton ) {
        constexpr uint8_t shift = sizeof(level)*8;
        key_t res = (morton << shift) + level; 
        assert( morton == (res >> shift) ); // Loss of data from shift
        return res;
    };

    KOKKOS_INLINE_FUNCTION uint32_t get_ioct_local(const OctantIndex& oct) const
    {
        return oct.isGhost*numOctants + oct.iOct;
    }

    level_t max_level;
    int ndim;

    // Fetches data from pmesh
    void init(std::shared_ptr<AMRmesh> pmesh, const HydroParams& params)
    {   
        oct_data_t::HostMirror oct_data_host("LightOctree::oct_data_host", pmesh->getNumOctants()+pmesh->getNumGhosts(), OCT_DATA_COUNT);
        oct_map_t::HostMirror oct_map_host( pmesh->getNumOctants()+pmesh->getNumGhosts());

        LightOctree_pablo mesh_pablo(pmesh, params);

        // Insert Octant in the map for his level with the morton index at this level as key
        auto add_octant = [&](const OctantIndex& oct)
        {
            pos_t c = mesh_pablo.getCorner(oct);
            uint8_t level = mesh_pablo.getLevel(oct);

            uint32_t ioct_local = get_ioct_local(oct);
            oct_data_host(ioct_local, ICORNERX) = c[IX];
            oct_data_host(ioct_local, ICORNERY) = c[IY];
            oct_data_host(ioct_local, ICORNERZ) = c[IZ];
            oct_data_host(ioct_local, ILEVEL) = level;

            uint32_t octant_count = std::pow( 2, level );
            real_t octant_size = 1.0/octant_count;
            real_t eps = octant_size/8; // To avoid rounding error when computing logical coords
            auto periodic_coord = [&](real_t pos) -> int32_t
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

            oct_map_t::insert_result inserted = oct_map_host.insert( get_key(level, morton), oct );
            assert(inserted.success());
        };

        for(uint32_t iOct = 0; iOct < pmesh->getNumOctants(); iOct++)
        {
            add_octant({iOct, false});
        }

        for(uint32_t iOct = 0; iOct < pmesh->getNumGhosts(); iOct++)
        {
            add_octant({iOct, true});
        }

        Kokkos::deep_copy(oct_map,oct_map_host);
        Kokkos::deep_copy(oct_data,oct_data_host);
    }
};

using LightOctree = LightOctree_hashmap;
//using LightOctree = LightOctree_pablo;

} //namespace dyablo
} //namespace muscl_block