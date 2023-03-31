#ifdef DYABLO_COMPILE_PABLO

#pragma once

#include "bitpit_PABLO.hpp"
#include "amr/UserDataLB.h"
#include "utils/mpi/MpiComm.h"

namespace dyablo {

class UserData;
class LightOctree_pablo;

/**
 * AMR mesh using PABLO as backend
 * This uses the PabloUniform interface + some methods to access ghost octants
 **/
class AMRmesh_pablo : private bitpit::PabloUniform
{
private:
    MpiComm mpi_comm;
    int level_min;
public:
    using bitpit::PabloUniform::getBordersPerProc;
    using bitpit::PabloUniform::getDim;
    using bitpit::PabloUniform::getNumOctants;
    using bitpit::PabloUniform::getNumGhosts;
    using bitpit::PabloUniform::getBound;
    using bitpit::PabloUniform::getCenter;
    using bitpit::PabloUniform::getCoordinates;
    using bitpit::PabloUniform::getLevel;
    using bitpit::PabloUniform::getGlobalNumOctants;
    using bitpit::PabloUniform::getGlobalIdx;
    using bitpit::PabloUniform::setMarker;
    using bitpit::PabloUniform::check21Balance;
    using bitpit::PabloUniform::checkToAdapt;

    const bitpit::PabloUniform& getPabloUniform() const
    {
        return *this;
    }

    bitpit::PabloUniform& getPabloUniform()
    {
        return *this;
    }

    AMRmesh_pablo( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max )
        : PabloUniform(dim), mpi_comm( PabloUniform::getComm() ), level_min(level_min)
    {
        assert(dim == 2 || dim == 3);
        assert(balance_codim <= dim);
        if( level_max > this->getMaxLevel() )
            throw std::runtime_error( std::string("level_max is too big for AMRmesh_pablo : is ") + std::to_string(level_max) + " but PABLO only supports " + std::to_string(this->getMaxLevel()) );
        bitpit::log::setConsoleVerbosity(this->getLog(), bitpit::log::Verbosity::QUIET);

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

        // Refine to level_min
        for (uint8_t level=0; level<level_min; ++level)
        {
            this->adaptGlobalRefine(); 
        } 
        this->loadBalance(0);
    }

    explicit AMRmesh_pablo( int dim )
     : AMRmesh_pablo(dim, 1, {false, false, false}, 1, 20)
    {}

    const MpiComm& getMpiComm() const
    {
        return mpi_comm;
    }

    std::array<bool, 6> getPeriodic() const{
        bitpit::bvector p = PabloUniform::getPeriodic();        
        if( p.size() == 4 )
            return {p[0], p[1], p[2], p[3], false, false};
        else //if( p.size() == 6 )
            return {p[0], p[1], p[2], p[3], p[4], p[5]};
    }  
    using PabloUniform::getPeriodic;

    int get_level_min() const
    {
        return level_min;
    }

    int get_max_supported_level()
    {
        return getMaxLevel()-1; // max_level=20 is too much for AMRmesh_pablo
    }

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

    bitpit::darray3 getSize( uint32_t iOct ) const
    {
        real_t size = ParaTree::getSize(iOct);
        return {size,size,size};
    }

    bitpit::darray3 getSizeGhost( uint32_t iOct ) const
    {
        real_t size = ParaTree::getSize(getGhostOctant(iOct));
        return {size,size,size};
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

    void adapt(int dummy)
    {
        bitpit::PabloUniform::adapt();
        pmesh_epoch++;
    }

    void adaptGlobalRefine()
    {
        bitpit::PabloUniform::adaptGlobalRefine();
        pmesh_epoch++;
    }

    void loadBalance( uint8_t compact_levels )
    {
#ifdef DYABLO_USE_MPI
        ParaTree::loadBalance(compact_levels);
        pmesh_epoch++;
#endif // DYABLO_USE_MPI
    }

    void loadBalance_userdata( uint8_t compact_levels, UserData& U );
protected:
    int pmesh_epoch=1;
};

} // namespace dyablo

#endif // DYABLO_COMPILE_PABLO
