#pragma once

#include "bitpit_PABLO.hpp"
#include "amr/UserDataLB.h"
#include "utils/mpi/MpiComm.h"

namespace dyablo {

/**
 * AMR mesh using PABLO as backend
 * This uses the PabloUniform interface + some methods to access ghost octants
 **/
class AMRmesh_pablo : public bitpit::PabloUniform
{
private:
    MpiComm mpi_comm;
public:
    AMRmesh_pablo( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max )
        : PabloUniform(dim), mpi_comm( PabloUniform::getComm() )
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

    void loadBalance( uint8_t compact_levels )
    {
#ifdef DYABLO_USE_MPI
        ParaTree::loadBalance(compact_levels);
#endif // DYABLO_USE_MPI
    }

    void loadBalance_userdata( uint8_t compact_levels, DataArrayBlock& U )
    {
        loadBalance_userdata_aux<DataArrayBlock, 2>(compact_levels, U);
    }

    void loadBalance_userdata( uint8_t compact_levels, DataArray& U )
    {
        loadBalance_userdata_aux<DataArray, 0>(compact_levels, U);
    }

private:
    template< typename DataArray_t, int iOct_pos >
    void loadBalance_userdata_aux( uint8_t compact_levels, DataArray_t& U )
    {
#ifdef DYABLO_USE_MPI
        using DataArrayHost_t = typename DataArray_t::HostMirror;
        // Copy Data to host for MPI communication 
        DataArrayHost_t U_host = Kokkos::create_mirror_view(U);
        Kokkos::deep_copy(U_host, U);
        
        DataArrayHost_t Ughost_host; // Dummy ghost array
        
        using UserDataLB_t = UserDataLB<DataArrayHost_t, iOct_pos> ;

        UserDataLB_t data_lb(U_host, Ughost_host);
        ParaTree::loadBalance<UserDataLB_t>(data_lb, compact_levels);

        Kokkos::realloc(U, U_host.layout());
        Kokkos::deep_copy(U, U_host);
#endif // DYABLO_USE_MPI
    }
};

} // namespace dyablo

