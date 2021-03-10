#pragma once

#include "shared/kokkos_shared.h"
#include "shared/amr/AMRmesh.h"

namespace dyablo{
namespace muscl_block{

/**
 * Interface to implement a ghost communicator
 **/
class GhostCommunicator_base
{
public:
    /**
     * Kokkos::View to be communicated by exchange_ghosts()
     * @tparam Datatype for Kokkos::view (e.g. real*** for amr-block user data)
     * View is LayoutLeft, and rightmost coordinate must be octant index (i.e. data for each octant is contiguous)
     **/
    //template <typename Datatype>
    //using View_t = Kokkos::View<Datatype,Kokkos::LayoutLeft>;

    /**
     * Exchange ghost octants data
     * @tparam View_DataType is Datatype template parameter for View_t
     *         (see View_t documentation). 
     * @param A array containing user data (read only)
     * @param A_ghost array to contain ghost data
     * Rightmost coordinate MUST be octant index (i.e. data for each octant is contiguous)
     * All contiguous data for each ghost octant is packed and sent using MPI
     **/
    //template< typename View_DataType >
    //void exchange_ghosts(const View_t<View_DataType>& A, View_t<View_DataType>& A_ghost) const;
};

/**
 * Ghost communicator that directly uses the mpi communication in PABLO
 **/
class GhostCommunicator_pablo : public GhostCommunicator_base
{
public:
    GhostCommunicator_pablo( std::shared_ptr<AMRmesh> amr_mesh )
        : amr_mesh(amr_mesh)
    {}
    
     /**
     * @copydoc GhostCommunicator_base::exchange_ghosts
     * 
     * Copy data back to host and call Paratree::communicate()
     **/
    void exchange_ghosts(const DataArrayBlock& U, DataArrayBlock& Ughost) const;
private:
    std::shared_ptr<AMRmesh> amr_mesh;  
};

/**
 * Ghost communicator that extracts communication metadata from PABLO 
 * then serialize/deserialize in Kokkos kernels and use CUDA-aware MPI 
 **/
class GhostCommunicator_kokkos : public GhostCommunicator_base
{
public:
    GhostCommunicator_kokkos( const std::map<int, std::vector<uint32_t>>& ghost_map );
    GhostCommunicator_kokkos( std::shared_ptr<AMRmesh> amr_mesh )
     : GhostCommunicator_kokkos(amr_mesh->getBordersPerProc())
    {}

    GhostCommunicator_kokkos( std::shared_ptr<AMRmesh_hashmap> amr_mesh )
     : GhostCommunicator_kokkos(amr_mesh->getBordersPerProc())
    {}
    
    /**
     * @copydoc GhostCommunicator_base::exchange_ghosts
     * 
     * Copy data back to host and call Paratree::communicate()
     **/
    void exchange_ghosts(const DataArrayBlock& U, DataArrayBlock& Ughost) const;
    void exchange_ghosts(const Kokkos::View<uint16_t**, Kokkos::LayoutLeft>& U, Kokkos::View<uint16_t**, Kokkos::LayoutLeft>& Ughost) const;
    void exchange_ghosts(const Kokkos::View<int*, Kokkos::LayoutLeft>& U, Kokkos::View<int*, Kokkos::LayoutLeft>& Ughost) const;
private:
    Kokkos::View<uint32_t*> recv_sizes, send_sizes; //!Number of octants to send/recv for each proc
    Kokkos::View<uint32_t*>::HostMirror recv_sizes_host, send_sizes_host; //!Number of octants to send/recv for each proc
    Kokkos::View<uint32_t*> send_iOcts; //! List of octants to send (first send_sizes[0] iOcts to send to rank[0] and so on...)
    uint32_t nbghosts_recv;

public:
    template< typename DataArray_t >
    void exchange_ghosts_aux( const DataArray_t& U, DataArray_t& Ughost) const;
};

using GhostCommunicator = GhostCommunicator_kokkos;
//using GhostCommunicator = GhostCommunicator_pablo;


}//namespace muscl_block
}//namespace dyablo