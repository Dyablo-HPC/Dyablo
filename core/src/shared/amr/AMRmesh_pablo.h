#pragma once

#include "bitpit_PABLO.hpp"

namespace dyablo {

/**
 * AMR mesh using PABLO as backend
 * This uses the PabloUniform interface + some methods to access ghost octants
 **/
class AMRmesh_pablo : public bitpit::PabloUniform
{
public:
    using bitpit::PabloUniform::PabloUniform;

    uint32_t getNodesCount(){
        return bitpit::ParaTree::getNodes().size();
    }

    double getXghost( uint32_t iOct ) const
    {
        return getX(getGhostOctant(iOct));
    }

    double getYghost( uint32_t iOct ) const
    {
        return getY(getGhostOctant(iOct));
    }

    double getZghost( uint32_t iOct ) const
    {
        return getZ(getGhostOctant(iOct));
    }

    double getSizeGhost( uint32_t iOct ) const
    {
        return getSize(getGhostOctant(iOct));
    }

    bitpit::darray3 getCenterGhost( uint32_t iOct ) const 
    {
        return getCenter(getGhostOctant(iOct));
    }

    bitpit::darray3 getCoordinatesGhost( uint32_t iOct ) const 
    {
        return getCoordinates(getGhostOctant(iOct));
    }

    uint8_t getLevelGhost( uint32_t iOct ) const 
    {
        return getLevel(getGhostOctant(iOct));
    }
};

} // namespace dyablo

