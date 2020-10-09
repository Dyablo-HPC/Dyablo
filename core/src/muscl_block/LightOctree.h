#pragma once

#include <memory>

#include "shared/kokkos_shared.h"
#include "shared/bitpit_common.h"

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

class LightOctree{
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

        uint8_t size() const
        {
            return m_size;
        }
        const OctantIndex& operator[](uint8_t i) const
        {
            return m_neighbors[i];
        }
    };
    

    LightOctree( std::shared_ptr<AMRmesh> pmesh )
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

        NeighborList neighbors;
        neighbors.m_size = iOct_neighbors.size();

        for(int i=0; i<neighbors.m_size; i++)
        {
            neighbors.m_neighbors[i].iOct = iOct_neighbors[i];
            neighbors.m_neighbors[i].isGhost = isghost_neighbors[i];
        }

        return neighbors;
    }

private:
    std::shared_ptr<AMRmesh> pmesh;
    uint8_t ndim;
};

} //namespace dyablo
} //namespace muscl_block