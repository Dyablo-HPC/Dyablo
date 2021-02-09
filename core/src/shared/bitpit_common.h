#ifndef BITPIT_COMMON_H_
#define BITPIT_COMMON_H_

#include "bitpit_PABLO.hpp"

#include "utils/mpiUtils/GlobalMpiSession.h"

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

    bitpit::darray3 getCoordinatesGhost( uint32_t iOct ) const 
    {
        return getCoordinates(getGhostOctant(iOct));
    }

    uint8_t getLevelGhost( uint32_t iOct ) const 
    {
        return getLevel(getGhostOctant(iOct));
    }

    // /** Get the logical coordinates of the nodes
    //  * \return Constant reference to the nodes matrix [nnodes*3] with the coordinates of the nodes.
    //  */
    // const bitpit::LocalTree::u32arr3vector &
    // getNodes() const {
    //     return this->getNodes();
    // }
    
};

class AMRmesh_hashmap : protected PABLO_mesh
{
private: 
    uint8_t dim;
    std::array<bool,3> periodic;

    uint64_t local_octs_count;
    enum octs_coord_id{
        IX, IY, IZ,
        LEVEL,
        NUM_OCTS_COORDS
    };
    using oct_view_t = Kokkos::View< uint16_t*[NUM_OCTS_COORDS] > ;
    oct_view_t local_octs_coord;

public:
    uint8_t getDim() const
    {
        return dim;
    }
    bitpit::bvector getPeriodic() const
    {
        return {periodic[0],periodic[0],periodic[1],periodic[1],periodic[2],periodic[2]};
    }
    bool getPeriodic(uint8_t i) const
    {
        assert(i<2*dim);
        return periodic[i/2];
    }
    //using PABLO_mesh::getDim;
    //using PABLO_mesh::getPeriodic;
    
    //MPI
    int getRank() const
    {
        return hydroSimu::GlobalMpiSession::getRank();
    }
    int getNproc() const
    {
        return hydroSimu::GlobalMpiSession::getNProc();
    }
    MPI_Comm getComm() const
    {
        return MPI_COMM_WORLD;
    }

    template< typename T >
    void communicate(T&)
    {
        assert(false); // communicate() cannot be used without PABLO
    }
    
    void loadBalance()
    {
        //TODO
        if( getNproc()>1 )
            assert(false); // Loadbalance cannot run in parallel yet
    }
    template< typename T >
    void loadBalance(T&, uint8_t level)
    {
        loadBalance();
    }

    const std::map<int, bitpit::u32vector>& getBordersPerProc() const
    {
        static std::map<int, bitpit::u32vector> dummy;
        if( getNproc()>1 )
            assert(false); // getBordersPerProc cannot run in parallel yet
        return dummy;
    }

    //using PABLO_mesh::getRank;
    //using PABLO_mesh::getNproc;
    //using PABLO_mesh::getComm;
    //using PABLO_mesh::loadBalance;
    //using PABLO_mesh::communicate;
    //using PABLO_mesh::getBordersPerProc;

    uint32_t getNumOctants() const
    {
        return local_octs_count;
    }

    uint32_t getNumGhosts() const
    {
        if( getNproc()>1 )
            assert(false); // getNumGhosts cannot run in parallel yet
        return 0;
    }

    uint32_t getGlobalNumOctants() const
    {
        if( getNproc()>1 )
            assert(false); // getGlobalNumOctants cannot run in parallel yet
        return local_octs_count;
    }

    //Get global state of mesh
    //using PABLO_mesh::getNumOctants;   
    //using PABLO_mesh::getNumGhosts;   
    //using PABLO_mesh::getGlobalNumOctants;   
    
    
    bool getBound( uint32_t idx ) const
    {
        uint16_t ix = local_octs_coord(idx, IX);
        uint16_t iy = local_octs_coord(idx, IY);
        uint16_t iz = local_octs_coord(idx, IZ);
        uint16_t level = getLevel(idx);

        uint32_t last_oct = std::pow(2, level)-1;

        return ix == 0 || iy == 0 || iz == 0 || ix == last_oct || iy == last_oct || iz == last_oct; 
    }

    std::array<real_t, 3> getCenter( uint32_t idx ) const
    {
        std::array<real_t, 3> corner = getCoordinates(idx);
        real_t size = getSize(idx);
        return { corner[IX]+size/2, corner[IY]+size/2, corner[IZ]+size/2 };
    }  

    std::array<real_t, 3> getCoordinates( uint32_t idx ) const
    {
        uint16_t ix = local_octs_coord(idx, IX);
        uint16_t iy = local_octs_coord(idx, IY);
        uint16_t iz = local_octs_coord(idx, IZ);

        real_t size = getSize(idx);

        return {ix*size, iy*size, iz*size};
    }

    real_t getSize( uint32_t idx ) const
    {
        uint16_t level = getLevel(idx);

        return 1.0/std::pow(2,level);
    }

    uint8_t getLevel( uint32_t idx ) const
    {
        return local_octs_coord(idx, LEVEL);
    }

    std::array<real_t, 3> getCenterGhost( uint32_t idx ) const
    {
        assert(false); // getCenterGhost cannot run in parallel yet
    }  

    std::array<real_t, 3> getCoordinatesGhost( uint32_t idx ) const
    {
        assert(false); // getCoordinatesGhost cannot run in parallel yet
    }

    real_t getSizeGhost( uint32_t idx ) const
    {
        assert(false); // getSizeGhost cannot run in parallel yet
    }

    uint8_t getLevelGhost( uint32_t idx ) const
    {
        assert(false); // getLevelGhost cannot run in parallel yet
    }

    uint32_t getGlobalIdx( uint32_t idx ) const
    {
        if( getNproc()>1 )
            assert(false); // getGlobalIdx cannot run in parallel yet
        return idx;
    }
    
    //Get octant info
    //using PABLO_mesh::getBound;
    //using PABLO_mesh::getCenter;
    //using PABLO_mesh::getCoordinates;
    //using PABLO_mesh::getLevel;
    //using PABLO_mesh::getSize;
    //using PABLO_mesh::getCenterGhost;
    //using PABLO_mesh::getCoordinatesGhost;
    //using PABLO_mesh::getSizeGhost;
    //using PABLO_mesh::getLevelGhost;
    //using PABLO_mesh::getGlobalIdx;
    
    bool getIsNewC(uint32_t idx) const
    {
        assert(false); //getIsNewC cannot be used without PABLO
        return false;
    }

    bool getIsNewR(uint32_t idx) const
    {
        assert(false); //getIsNewR cannot be used without PABLO
        return false;
    }

    void getMapping(uint32_t & idx, std::vector<uint32_t> & mapper, std::vector<bool> & isghost) const
    {
        assert(false); //getMapping cannot be used without PABLO
    }

    void adaptGlobalRefine()
    {
        #error "todo"
    }

    void setMarker(uint32_t iOct, int marker)
    {
        //TODO
    }

    void adapt(bool dummy = true)
    {
        //TODO
    }

    // Adapt
    //using PABLO_mesh::adaptGlobalRefine;
    //using PABLO_mesh::setMarker;
    //using PABLO_mesh::adapt;
    // using PABLO_mesh::getIsNewC;
    // using PABLO_mesh::getIsNewR;
    // using PABLO_mesh::getMapping;
    //Debug only
    //using PABLO_mesh::check21Balance;
    //using PABLO_mesh::checkToAdapt;

    bool check21Balance()
    {
        //TODO
        return true;
    }

    bool checkToAdapt()
    {
        //TODO
        return false;
    }

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
    AMRmesh_hashmap( int dim, int balance_codim, const std::array<bool,3>& periodic )
        : PABLO_mesh(dim), dim(dim), periodic(periodic)
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

        this->local_octs_count = 1;
        this->local_octs_coord = oct_view_t("local_octs_coord", 1);
    }
    
    void findNeighbours(uint32_t iOct, uint8_t iface, uint8_t codim , 
                        bitpit::u32vector& neighbor_iOcts, bitpit::bvector& neighbor_isGhost) const
    {
        assert(false); // findneighbours() cannot be used without PABLO
    }



};

using AMRmesh = AMRmesh_hashmap;

} // namespace dyablo

#endif // BITPIT_COMMON_H_
