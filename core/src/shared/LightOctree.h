#pragma once

#include <memory>
#include <unordered_map>

#include "shared/morton_utils.h"
#include "shared/kokkos_shared.h"
#include "shared/bitpit_common.h"
#include "shared/HydroParams.h"
#include "Kokkos_UnorderedMap.hpp"
#include "Kokkos_Macros.hpp"

namespace dyablo { 

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

/**
 * Interface for a simplified read-only octree that can be accessed to get octant information
 * (position, amr level, neighbors, ...)
 * 
 * Class containing types and methods to implement a LightOctree
 * LightOctree implementations implement this interface, but don't necessarily derive from it 
 * (this base class is not polymorphic!).
 **/
class LightOctree_base{
public:
    
    /// Index to a PABLO octant
    struct OctantIndex
    {
        uint32_t iOct; //! PABLO's Octant index
        bool isGhost; //! Is this a MPI ghost octant?
    };
    /// Physical cell position
    using pos_t = Kokkos::Array<real_t,3>;
    /// Relative position of a neighbor octant relative to local octant
    using offset_t = Kokkos::Array<int8_t,3>;
    /// Container for 0-4 neighbor(s)
    struct NeighborList
    {
        uint8_t m_size;
        Kokkos::Array<OctantIndex,4> m_neighbors;
        /// Number of neighbors in container (0-4)
        KOKKOS_INLINE_FUNCTION uint8_t size() const
        {
            return m_size;
        }
        /// Get i-th neighbor index in container
        KOKKOS_INLINE_FUNCTION const OctantIndex& operator[](uint8_t i) const
        {
            return m_neighbors[i];
        }
    };
    /// Get local (MPI) octant count
    uint32_t getNumOctants() const;
    //bool getBound(const OctantIndex& iOct)  const;
    /// Get physical position of Octant center
    pos_t getCenter(const OctantIndex& iOct)  const;
    /// Get physical position of octant corner (smallest position inside octant)
    pos_t getCorner(const OctantIndex& iOct)  const;
    /// Get physical size of octant in all dimensions (Octant is a cube)
    real_t getSize(const OctantIndex& iOct)  const;
    /// Get amr level of octant
    uint8_t getLevel(const OctantIndex& iOct)  const;
    /**
     * Get neighbors of `iOct` at relative position `offset`
     * 
     * @param iOct local Octant index (cannot be a ghost octant)
     * @param offset Relative position of neighbor(s) to fetch.
     *               offset in each dimension is either -1, 0 or 1; {0,0,0} is invalid
     *               in 2D, third dimension is always 0
     *               ex : {-1,0,0} is left neighbor; {-1,-1,0} is lower-left edge(3D)/corner(2D) 
     *
     * @returns between 1 and 4 neighbors packed in a NeighborList
     * 
     * ex in 2D:
     * ```
     *   ___________ __________
     *  |           |     |    |
     *  |           |  14 | 15 |
     *  |    11     |-----+----|
     *  |           |  12 | 13 |
     *  |___________|_____|____|
     *  |     |     | 8|9 |    |
     *  |  2  |  3  | 6|7 | 10 |
     *  |-----+-----|-----+----|
     *  |  0  |  1  |  4  | 5  |
     *  |_____|_____|_____|____|
     * 
     * findNeighbors({8,true},{-1, 1, 0}) -> 11
     * findNeighbors({8,true},{-1, 0, 0}) -> 3
     * findNeighbors({8,true},{-1,-1, 0}) -> 3 (Note that same neighbor can be returned twice)
     * findNeighbors({3,true},{ 1, 0, 0}) -> {8,6}
     * findNeighbors({3,true},{ 1, 1, 0}) -> 12
     * 
     * findNeighbors({1,true},{ 1, 1, 0}) -> 6 (Note that there is only 1 smaller neighbor in corners)
     * ```
     *
     * @note findNeighbor(), unlike PABLO's, always returns all neighbors in corner 
     * @note Requesting a neighbor outside the domain when PABLO octree is not periodic is undefined behavior
     **/
    NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset ) const;
};

/**
 * Implementation of the LightOctree_base interface based on PABLO
 * 
 * This implementation calls PABLO to find neighbors and get octant data.
 * It can't be used on device code with the Kokkos::CUDA backend.
 * 
 **/ 
class LightOctree_pablo : public LightOctree_base{
public:
    LightOctree_pablo() = default;
    LightOctree_pablo(const LightOctree_pablo& ) = default;


    LightOctree_pablo( std::shared_ptr<AMRmesh> pmesh, const HydroParams& params )
    : pmesh(pmesh), ndim(pmesh->getDim())
    {}
    //! @copydoc LightOctree_base::getNumOctants()
    uint32_t getNumOctants() const
    {
        return pmesh->getNumOctants();
    }
    //! @copydoc LightOctree_base::getNdim()
    KOKKOS_INLINE_FUNCTION uint8_t getNdim() const
    {
        return ndim;
    }
    //! @copydoc LightOctree_base::getBound()
    bool getBound(const OctantIndex& iOct)  const
    {
        assert( !iOct.isGhost );
        return pmesh->getBound(iOct.iOct);
    }
    //! @copydoc LightOctree_base::getCenter()
    pos_t getCenter(const OctantIndex& iOct)  const
    {
        bitpit::darray3 pcenter = iOct.isGhost ? pmesh->getCenter(pmesh->getGhostOctant(iOct.iOct)) : pmesh->getCenter(iOct.iOct);
        return {pcenter[0], pcenter[1], pcenter[2]};
    }
    //! @copydoc LightOctree_base::getCorner()
    pos_t getCorner(const OctantIndex& iOct)  const
    {
        bitpit::darray3 pmin = iOct.isGhost ? 
                pmesh->getCoordinates(pmesh->getGhostOctant(iOct.iOct)) : 
                pmesh->getCoordinates(iOct.iOct);
        return {pmin[IX], pmin[1], pmin[2]};
    }
    //! @copydoc LightOctree_base::getSize()
    real_t getSize(const OctantIndex& iOct)  const
    {
        real_t oct_size = iOct.isGhost ? 
                pmesh->getSize(pmesh->getGhostOctant(iOct.iOct)) : 
                pmesh->getSize(iOct.iOct);
        return oct_size;
    }
    //! @copydoc LightOctree_base::getLevel()
    uint8_t getLevel(const OctantIndex& iOct)  const
    {
        uint8_t oct_level = iOct.isGhost ? 
                pmesh->getLevel(pmesh->getGhostOctant(iOct.iOct)) : 
                pmesh->getLevel(iOct.iOct);
        return oct_level;
    }
    /**
     * @copydoc LightOctree_base::findNeighbors()
     * @note findNeighbors() may call PABLO's findNeighbours() multiple times 
     *       to correctly get neighbors in corners
     **/
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

        //Find neighbors with PABLO
        std::vector<uint32_t> iOct_neighbors;
        std::vector<bool> isghost_neighbors;
        pmesh->findNeighbours(iOct.iOct, pablo_iface, pablo_codim, iOct_neighbors, isghost_neighbors);

        // Fill NeighborList from PABLO's result
        NeighborList neighbors={0,{99,true}};
        if( iOct_neighbors.size() == 0 )
        {
            // If PABLO returns 0 neighbors, he might be wrong (see fix_missing_corner_neighbor())
            pos_t cellPos = getCenter(iOct);
            uint8_t level = getLevel(iOct);
            real_t cellSize = 1.0/std::pow(2, level );
            cellPos[IX] += offset[IX]*cellSize*0.6;
            cellPos[IY] += offset[IY]*cellSize*0.6;
            cellPos[IZ] += offset[IZ]*cellSize*0.6;
            bitpit::bvector periodic = pmesh->getPeriodic();
            // Maybe really no neighbor if outside domain
            if( ( periodic[2*IX] or ( 0.0 <= cellPos[IX] && cellPos[IX] < 1.0 ) )
             and ( periodic[2*IY] or ( 0.0 <= cellPos[IY] && cellPos[IY] < 1.0 ) )
             and ( periodic[2*IZ] or ( 0.0 <= cellPos[IZ] && cellPos[IZ] < 1.0 ) ) )
            {
                //Get periodic position inside domain
                if(periodic[2*IX]) cellPos[IX] -= std::floor(cellPos[IX]/1.0);
                if(periodic[2*IY]) cellPos[IY] -= std::floor(cellPos[IY]/1.0);
                if(periodic[2*IZ]) cellPos[IZ] -= std::floor(cellPos[IZ]/1.0);

                fix_missing_corner_neighbor(iOct.iOct, offset, cellPos, neighbors);
            }
            else
            {
                assert(false); // This is undefined behavior ( see LightOctree_base::findNeighbors doc )
            }
        }
        else
        {
            // Use PABLO's result when 
            neighbors.m_size = iOct_neighbors.size();
            for(int i=0; i<neighbors.m_size; i++)
            {
                neighbors.m_neighbors[i].iOct = iOct_neighbors[i];
                neighbors.m_neighbors[i].isGhost = isghost_neighbors[i];
            }
        }

        return neighbors;
    }

    // ------------------------
    // Only in LightOctree_pablo
    // ------------------------
    std::shared_ptr<AMRmesh> getMesh() const{
        return pmesh;
    }

protected:
    std::shared_ptr<AMRmesh> pmesh; //! PABLO mesh to relay requests to
    uint8_t ndim; //! 2D or 3D

private:
    /** 
     * When neighbor is larger, it is returned in only one `findNeighbours()` request
     * Sometimes `findNeighbours()` edge/node returns 0 neighbors because the only neighbor has already been returned by findNeighbour
     * on a lower codimension request. In this case, actual edge/node neighbor has to be searched in lower codimention neighbors
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
    LightOctree_hashmap() = default;
    LightOctree_hashmap(const LightOctree_hashmap& lmesh) = default;

    LightOctree_hashmap( std::shared_ptr<AMRmesh> pmesh, const HydroParams& params )
    : oct_map(pmesh->getNumOctants()+pmesh->getNumGhosts()),
      oct_data("LightOctree::oct_data", pmesh->getNumOctants()+pmesh->getNumGhosts(), OCT_DATA_COUNT),
      numOctants(pmesh->getNumOctants()) , min_level(params.level_min), max_level(params.level_max), ndim(pmesh->getDim())
    {
        std::cout << "LightOctree rehash ..." << std::endl;
        init(pmesh, params);
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
    // bool getBound(const OctantIndex& iOct)  const
    // {
    //     assert( !iOct.isGhost );
    //     return pmesh->getBound(iOct.iOct);
    // }
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

    // ------------------------
    // Only in LightOctree_hashmap
    // ------------------------
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
        return oct.isGhost*numOctants + oct.iOct;
    }

    int numOctants; //! Number of local octants (no ghosts)
    level_t min_level; //! Coarser level of the octree
    level_t max_level; //! Finer level of the octree
    int ndim; //! 2D or 3D
    

    /**
     * Fetches data from pmesh and fill hashmap
     **/
    void init(std::shared_ptr<AMRmesh> pmesh, const HydroParams& params)
    {   
        oct_data_t::HostMirror oct_data_host("LightOctree::oct_data_host", pmesh->getNumOctants()+pmesh->getNumGhosts(), OCT_DATA_COUNT);
        oct_map_t::HostMirror oct_map_host( pmesh->getNumOctants()+pmesh->getNumGhosts());

        LightOctree_pablo mesh_pablo(pmesh, params);

        // Get octant data using LightOctree_pablo and 
        // insert Octant in oct_data_host and oct_map_host
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
        // Insert local octants
        for(uint32_t iOct = 0; iOct < pmesh->getNumOctants(); iOct++)
        {
            add_octant({iOct, false});
        }
        // Inser ghost octants
        for(uint32_t iOct = 0; iOct < pmesh->getNumGhosts(); iOct++)
        {
            add_octant({iOct, true});
        }
        // Copy data and hashmap to device
        Kokkos::deep_copy(oct_map,oct_map_host);
        Kokkos::deep_copy(oct_data,oct_data_host);
    }
};

#ifdef KOKKOS_ENABLE_CUDA
using LightOctree = LightOctree_hashmap;
#else
using LightOctree = LightOctree_pablo;
#endif

} //namespace dyablo