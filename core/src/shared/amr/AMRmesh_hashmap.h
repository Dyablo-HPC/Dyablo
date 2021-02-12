#pragma once

#include <vector>
#include "mpi.h"
#include "Kokkos_UnorderedMap.hpp"

#include "utils/mpiUtils/GlobalMpiSession.h"
#include "shared/kokkos_shared.h"

namespace dyablo {

/**
 * AMR mesh without PABLO 
 * This uses the PabloUniform interface + some methods to access ghost octants
 **/
class AMRmesh_hashmap
{
public:
    enum octs_coord_id{
        IX, IY, IZ,
        LEVEL,
        NUM_OCTS_COORDS
    };
    using oct_view_t = Kokkos::View< uint16_t*[NUM_OCTS_COORDS] > ;
    using markers_t = Kokkos::UnorderedMap<uint32_t, int>;

private: 
    uint8_t dim;
    std::array<bool,3> periodic;

    uint32_t local_octs_count;
    oct_view_t local_octs_coord;
    markers_t markers;    

    uint8_t level_min, level_max;

public:
    uint8_t getDim() const
    {
        return dim;
    }

    std::vector<bool> getPeriodic() const
    {
        return {periodic[0],periodic[0],periodic[1],periodic[1],periodic[2],periodic[2]};
    }

    bool getPeriodic(uint8_t i) const
    {
        assert(i<2*dim);
        return periodic[i/2];
    }

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

    const std::map<int, std::vector<uint32_t>>& getBordersPerProc() const
    {
        static std::map<int, std::vector<uint32_t>> dummy;
        if( getNproc()>1 )
            assert(false); // getBordersPerProc cannot run in parallel yet
        return dummy;
    }

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

    void adaptGlobalRefine();
    void setMarkersCapacity(uint32_t capa);
    void setMarker(uint32_t iOct, int marker);
    void adapt(bool dummy = true);

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
    void computeConnectivity(){}
    void updateConnectivity(){}

    /**
     * @param dim number of dimensions 2D/3D
     * @param balance_codim 2:1 balance behavior : 
     *               1 ==> balance through faces, 
     *               2 ==> balance through faces and corner
     *               3 ==> balance through faces, edges and corner (3D only)
     * @param periodic set perodicity for each dimension (last is ignored in 2D)
     **/
    AMRmesh_hashmap( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max);
    
    void findNeighbours(uint32_t iOct, uint8_t iface, uint8_t codim , 
                        std::vector<uint32_t>& neighbor_iOcts, std::vector<bool>& neighbor_isGhost) const
    {
        assert(false); // findneighbours() cannot be used without PABLO
    }
};

} // namespace dyablo

