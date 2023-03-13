#pragma once

#include <vector>
#include <map>
#include "Kokkos_UnorderedMap.hpp"

#include "utils/mpi/GlobalMpiSession.h"
#include "kokkos_shared.h"

namespace dyablo {

class UserData;

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
    using coord_t = uint32_t;
    using level_t = uint16_t;
    using oct_view_device_t = Kokkos::View<coord_t**, Kokkos::LayoutLeft> ;
    using markers_device_t = Kokkos::UnorderedMap<uint32_t, int>;
    using oct_view_t = oct_view_device_t::HostMirror ;
    using markers_t = markers_device_t::HostMirror;

private: 
    uint8_t dim;
    std::array<bool,3> periodic;

    uint64_t total_octs_count; /// Total number of octs across all MPIs 
    uint64_t global_id_begin; /// Global index of first local octant (number of octs in lower rank MPIs)
    
    oct_view_t local_octs_coord; /// Local octants data
    oct_view_t ghost_octs_coord; /// Ghost Octants data
    markers_t markers;

    bool sequential_mesh = true; /// Mesh is sequential before the first loadbalance() call

    std::map<int, std::vector<uint32_t>> local_octants_to_send; /// rank -> array of remote ghosts to send to rank

    level_t level_min, level_max;
    MpiComm mpi_comm;

public:
    uint8_t getDim() const
    {
        return dim;
    }

    std::array<bool, 6> getPeriodic() const
    {
        return {periodic[0],periodic[0],periodic[1],periodic[1],periodic[2],periodic[2]};
    }

    int get_level_min() const
    {
        return level_min;
    }

    int get_max_supported_level()
    {
        return sizeof(oct_view_device_t::value_type)*8-1;
    }

    bool getPeriodic(uint8_t i) const
    {
        assert(i<6);
        return periodic[i/2];
    }

    //MPI
    const MpiComm& getMpiComm() const
    {
        return mpi_comm;
    }
    int getRank() const
    {
        return mpi_comm.MPI_Comm_rank();
    }
    int getNproc() const
    {
        return mpi_comm.MPI_Comm_size();
    }
    
    /**
     * \brief Change octants distribution to redistribute the load
     * \return a map contianig exchanged octants to create 
     *         a GhostCommunicator_kokkos to communicate user data of migrated octants
     **/
    
    std::map<int, std::vector<uint32_t>> loadBalance(level_t level=0);

    template< typename T >
    void loadBalance(T&, level_t level)
    {
        if(this->getNproc() > 1)
            assert(false); // loadBalance( UserCommLB ) cannot be used without PABLO : User data are not exchanged
    }

    void loadBalance_userdata( int compact_levels, UserData& U );

    const std::map<int, std::vector<uint32_t>>& getBordersPerProc() const;

    uint32_t getNumOctants() const
    {
        return local_octs_coord.extent(1);
    }

    uint32_t getNumGhosts() const
    {
        return ghost_octs_coord.extent(1);
    }

    uint32_t getGlobalNumOctants() const
    {
        return total_octs_count;
    } 

    bool getBound( uint32_t idx ) const
    {
        coord_t ix = local_octs_coord(IX, idx);
        coord_t iy = local_octs_coord(IY, idx);
        coord_t iz = local_octs_coord(IZ, idx);
        level_t level = getLevel(idx);

        coord_t last_oct = std::pow(2, level)-1;

        return ix == 0 || iy == 0 || iz == 0 || ix == last_oct || iy == last_oct || iz == last_oct; 
    }

    std::array<real_t, 3> getCenter( uint32_t idx ) const
    {
        std::array<real_t, 3> corner = getCoordinates(idx);
        auto size = getSize(idx);
        return { corner[IX]+size[IX]/2, corner[IY]+size[IY]/2, corner[IZ]+size[IZ]/2 };
    }


    std::array<real_t, 3> getCoordinates( uint32_t idx ) const
    {
        coord_t ix = local_octs_coord(IX, idx);
        coord_t iy = local_octs_coord(IY, idx);
        coord_t iz = local_octs_coord(IZ, idx);

        auto size = getSize(idx);

        return {ix*size[IX], iy*size[IY], iz*size[IZ]};
    }

    std::array<real_t, 3> getSize( uint32_t idx ) const
    {
        level_t level = getLevel(idx);

        real_t size = 1.0/std::pow(2,level);
        return {size,size,size};
    }

    level_t getLevel( uint32_t idx ) const
    {
        return local_octs_coord(LEVEL, idx);
    }

    std::array<real_t, 3> getCenterGhost( uint32_t idx ) const
    {
        std::array<real_t, 3> corner = getCoordinatesGhost(idx);
        auto size = getSizeGhost(idx);
        return { corner[IX]+size[IX]/2, corner[IY]+size[IY]/2, corner[IZ]+size[IZ]/2 };
    }  

    std::array<real_t, 3> getCoordinatesGhost( uint32_t idx ) const
    {
        coord_t ix = ghost_octs_coord(IX, idx);
        coord_t iy = ghost_octs_coord(IY, idx);
        coord_t iz = ghost_octs_coord(IZ, idx);

        auto size = getSizeGhost(idx);

        return {ix*size[IX], iy*size[IY], iz*size[IZ]};
    }


    std::array<real_t, 3> getSizeGhost( uint32_t idx ) const
    {
        level_t level = getLevelGhost(idx);

        real_t size = 1.0/std::pow(2,level);
        return {size,size,size};
    }

    level_t getLevelGhost( uint32_t idx ) const
    {
        return ghost_octs_coord(LEVEL, idx);
    }

    uint32_t getGlobalIdx( uint32_t idx ) const
    {
        return global_id_begin+idx;
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
    AMRmesh_hashmap( int dim, int balance_codim, const std::array<bool,3>& periodic, level_t level_min, level_t level_max, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world());
    
    void findNeighbours(uint32_t iOct, uint8_t iface, uint8_t codim , 
                        std::vector<uint32_t>& neighbor_iOcts, std::vector<bool>& neighbor_isGhost) const
    {
        assert(false); // findneighbours() cannot be used without PABLO
    }
protected:
    int pmesh_epoch = 1;
};

} // namespace dyablo
