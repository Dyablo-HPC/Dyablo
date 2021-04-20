#pragma once

#include "shared/kokkos_shared.h"

namespace dyablo { 

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

        KOKKOS_INLINE_FUNCTION static uint32_t OctantIndex_to_iOctLocal(const OctantIndex& oct, uint32_t numOctants)
        {
            // Ghosts are stored after non-ghosts
            return oct.isGhost*numOctants + oct.iOct;
        }

        KOKKOS_INLINE_FUNCTION static OctantIndex iOctLocal_to_OctantIndex(uint32_t ioct_local, uint32_t numOctants)
        {
            OctantIndex oct = {ioct_local, false};
            if( ioct_local >= numOctants )
            {
                oct.iOct -= numOctants;
                oct.isGhost = true;
            }
            return oct;
        }
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
     * @note Requesting a neighbor outside the domain when PABLO octree is not periodic returns 
     *       an empty neighbor list (but you should use isBoundary() if you only want to test that)
     **/
    NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset ) const;

    /// Is the given face of the given oct an external boundary ?
    bool isBoundary(const OctantIndex& iOct, const offset_t& offset) const;

};

} //namespace dyablo
