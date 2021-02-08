#ifndef BITPIT_COMMON_H_
#define BITPIT_COMMON_H_

#include "bitpit_PABLO.hpp"

namespace dyablo {

//! bitpit Pablo object type alias
//using AMRmesh = bitpit::PabloUniform;

class PABLO_mesh : public bitpit::PabloUniform
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

class AMRmesh : protected PABLO_mesh
{
public:
    using PABLO_mesh::getDim;
    using PABLO_mesh::getPeriodic;
    //MPI
    using PABLO_mesh::getRank;
    using PABLO_mesh::getNproc;
    using PABLO_mesh::getComm;
    using PABLO_mesh::isCommSet;
    using PABLO_mesh::loadBalance;
    using PABLO_mesh::communicate;
    using PABLO_mesh::getBordersPerProc;
    
    //Get global state of mesh
    using PABLO_mesh::getNumOctants;   
    using PABLO_mesh::getNumGhosts;   
    using PABLO_mesh::getGlobalNumOctants;   
    //Get octant info
    using PABLO_mesh::getBound;
    using PABLO_mesh::getCenter;
    using PABLO_mesh::getCoordinates;
    //using PABLO_mesh::getX;
    //using PABLO_mesh::getY;
    //using PABLO_mesh::getZ;
    using PABLO_mesh::getLevel;
    using PABLO_mesh::getSize;
    using PABLO_mesh::getGhostOctant;
    using PABLO_mesh::getGlobalIdx;
    
    // Adapt
    using PABLO_mesh::setMarker;
    using PABLO_mesh::adaptGlobalRefine;
    using PABLO_mesh::adapt;
    using PABLO_mesh::getIsNewC;
    using PABLO_mesh::getIsNewR;
    using PABLO_mesh::getMapping;
    using PABLO_mesh::check21Balance;
    using PABLO_mesh::checkToAdapt;

    // Connectivity and nodes
    void computeConnectivity(){}
    void updateConnectivity(){}
    //using PABLO_mesh::getConnectivity;
    //using PABLO_mesh::getNnodes;
    //using PABLO_mesh::getNode;
    //using PABLO_mesh::getNodes;
    //using PABLO_mesh::getNodeCoordinates;

    //Misc
    //using PABLO_mesh::getLog;

    /**
     * @param dim number of dimensions 2D/3D
     * @param balance_codim 2:1 balance behavior : 
     *               1 ==> balance through faces, 
     *               2 ==> balance through faces and corner
     *               3 ==> balance through faces, edges and corner (3D only)
     * @param periodic set perodicity for each dimension (last is ignored in 2D)
     **/
    AMRmesh( int dim, int balance_codim, const std::array<bool,3>& periodic )
        : PABLO_mesh(dim)
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
    
    void findNeighbours(uint32_t iOct, uint8_t iface, uint8_t codim , 
                        bitpit::u32vector& neighbor_iOcts, bitpit::bvector& neighbor_isGhost) const
    {
        bitpit::PabloUniform::findNeighbours( iOct, iface, codim , neighbor_iOcts, neighbor_isGhost);
    }    
};

} // namespace dyablo

#endif // BITPIT_COMMON_H_
