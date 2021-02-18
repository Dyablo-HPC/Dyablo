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
    AMRmesh_pablo( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max )
        : PabloUniform(dim)
    {
        assert(dim == 2 || dim == 3);
        assert(balance_codim <= dim);

        this->setBalanceCodimension(balance_codim);
        uint32_t idx = 0;
        this->setBalance(idx,true);
        if(periodic[0])
        {
            this->setPeriodic(0);
            this->setPeriodic(1);
        }
        if(periodic[1])
        {
            this->setPeriodic(2);
            this->setPeriodic(3);
        }
        if(dim>=3 && periodic[2])
        {
            this->setPeriodic(4);
            this->setPeriodic(5);
        }
    }

    explicit AMRmesh_pablo( int dim )
     : AMRmesh_pablo(dim, 1, {false, false, false}, 1, 20)
    {}
    

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

    void setMarkersCapacity(uint32_t capa){}
};

} // namespace dyablo

