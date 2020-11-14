#ifndef BITPIT_COMMON_H_
#define BITPIT_COMMON_H_

#include "bitpit_PABLO.hpp"

namespace dyablo {

//! bitpit Pablo object type alias
//using AMRmesh = bitpit::PabloUniform;

class AMRmesh : public bitpit::PabloUniform
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

    // /** Get the logical coordinates of the nodes
    //  * \return Constant reference to the nodes matrix [nnodes*3] with the coordinates of the nodes.
    //  */
    // const bitpit::LocalTree::u32arr3vector &
    // getNodes() const {
    //     return this->getNodes();
    // }
    
};

} // namespace dyablo

#endif // BITPIT_COMMON_H_
