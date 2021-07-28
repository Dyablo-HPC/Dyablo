#pragma once

#include <memory>

#include "shared/amr/AMRmesh.h"
#include "shared/amr/LightOctree_base.h"

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

    template< typename AMRmesh_t >
    LightOctree_pablo( AMRmesh_t* pmesh, uint8_t level_min, uint8_t level_max )
     : LightOctree_pablo(pmesh->getMesh(), level_min, level_max)
    {
        static_assert( std::is_same<AMRmesh_t, AMRmesh_impl<AMRmesh_pablo>>::value, "LightOctree_pablo is only compatible with AMRmesh_pablo" );
    }

    LightOctree_pablo( AMRmesh_pablo& pmesh_, uint8_t level_min, uint8_t level_max )
    : pmesh(&pmesh_), ndim(pmesh->getDim())
    {
        is_periodic[IX] = pmesh->getPeriodic(IX*2);
        is_periodic[IY] = pmesh->getPeriodic(IY*2);
        is_periodic[IZ] = pmesh->getPeriodic(IZ*2);
    }
    //! @copydoc LightOctree_base::getNumOctants()
    uint32_t getNumOctants() const
    {
        return pmesh->getNumOctants();
    }
    //! @copydoc LightOctree_base::getNumOctants()
    uint32_t getNumGhosts() const
    {
        return pmesh->getNumGhosts();
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
        bitpit::darray3 pcenter = iOct.isGhost ? pmesh->getCenterGhost(iOct.iOct) : pmesh->getCenter(iOct.iOct);
        return {pcenter[0], pcenter[1], pcenter[2]};
    }
    //! @copydoc LightOctree_base::getCorner()
    pos_t getCorner(const OctantIndex& iOct)  const
    {
        bitpit::darray3 pmin = iOct.isGhost ? 
                pmesh->getCoordinatesGhost(iOct.iOct) : 
                pmesh->getCoordinates(iOct.iOct);
        return {pmin[IX], pmin[1], pmin[2]};
    }
    //! @copydoc LightOctree_base::getSize()
    real_t getSize(const OctantIndex& iOct)  const
    {
        real_t oct_size = iOct.isGhost ? 
                pmesh->getSizeGhost(iOct.iOct) : 
                pmesh->getSize(iOct.iOct);
        return oct_size;
    }
    //! @copydoc LightOctree_base::getLevel()
    uint8_t getLevel(const OctantIndex& iOct)  const
    {
        uint8_t oct_level = iOct.isGhost ? 
                pmesh->getLevelGhost(iOct.iOct) : 
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

        if( this->isBoundary(iOct, offset) )
            return NeighborList{0,{}};

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
            // Maybe really no neighbor if outside domain
            if( ( is_periodic[IX] or ( 0.0 <= cellPos[IX] && cellPos[IX] < 1.0 ) )
             and ( is_periodic[IY] or ( 0.0 <= cellPos[IY] && cellPos[IY] < 1.0 ) )
             and ( is_periodic[IZ] or ( 0.0 <= cellPos[IZ] && cellPos[IZ] < 1.0 ) ) )
            {
                //Get periodic position inside domain
                if(is_periodic[IX]) cellPos[IX] -= std::floor(cellPos[IX]/1.0);
                if(is_periodic[IY]) cellPos[IY] -= std::floor(cellPos[IY]/1.0);
                if(is_periodic[IZ]) cellPos[IZ] -= std::floor(cellPos[IZ]/1.0);

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
    /// @copydoc LightOctree_base::isBoundary()
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
          || (ndim==3 && !this->is_periodic[IZ] && !( 0<pos[IZ] && pos[IZ]<1 )) ;      
    }

    // ------------------------
    // Only in LightOctree_pablo
    // ------------------------
    AMRmesh_pablo* getMesh() const{
        return pmesh;
    }

protected:
    AMRmesh_pablo* pmesh; //! PABLO mesh to relay requests to
    uint8_t ndim; //! 2D or 3D
    Kokkos::Array<bool,3> is_periodic;

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

} //namespace dyablo
