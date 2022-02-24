#pragma once

#include "kokkos_shared.h"
#include "amr/AMRmesh.h"

namespace dyablo{


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
    template< typename AMRmesh_t >
    GhostCommunicator_pablo( std::shared_ptr<AMRmesh_t> amr_mesh )
        : amr_mesh(amr_mesh->getMesh())
    {}
    
     /**
     * @copydoc GhostCommunicator_base::exchange_ghosts
     * 
     * Copy data back to host and call Paratree::communicate()
     **/
    void exchange_ghosts(const DataArrayBlock& U, DataArrayBlock& Ughost) const;
    
    /// Note : octant index for DataArray is leftmost subscript
    void exchange_ghosts(const DataArray& U, DataArray& Ughost) const;
private:
    AMRmesh_pablo& amr_mesh;  
};

/**
 * Ghost communicator that extracts communication metadata from PABLO 
 * then serialize/deserialize in Kokkos kernels and use CUDA-aware MPI 
 **/
class GhostCommunicator_kokkos : public GhostCommunicator_base
{
public:
    GhostCommunicator_kokkos( const std::map<int, std::vector<uint32_t>>& ghost_map, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() );
    GhostCommunicator_kokkos( std::shared_ptr<AMRmesh> amr_mesh, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
     : GhostCommunicator_kokkos(amr_mesh->getBordersPerProc(), mpi_comm)
    {}

    GhostCommunicator_kokkos( std::shared_ptr<AMRmesh_hashmap> amr_mesh, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
     : GhostCommunicator_kokkos(amr_mesh->getBordersPerProc(), mpi_comm)
    {}
    
    /**
     * @copydoc GhostCommunicator_base::exchange_ghosts
     * 
     * Copy data back to host and call Paratree::communicate()
     **/
    void exchange_ghosts(const DataArrayBlock& U, DataArrayBlock& Ughost) const;
    void exchange_ghosts(const Kokkos::View<uint16_t**, Kokkos::LayoutLeft>& U, Kokkos::View<uint16_t**, Kokkos::LayoutLeft>& Ughost) const;
    void exchange_ghosts(const LightOctree_storage<>::oct_data_t& U, LightOctree_storage<>::oct_data_t& Ughost) const;
    void exchange_ghosts(const Kokkos::View<uint32_t**, Kokkos::LayoutLeft>& U, Kokkos::View<uint32_t**, Kokkos::LayoutLeft>& Ughost) const;
    void exchange_ghosts(const Kokkos::View<int*, Kokkos::LayoutLeft>& U, Kokkos::View<int*, Kokkos::LayoutLeft>& Ughost) const;

    /// note : iOct_pos is 0 for DataArray. (See exchange_ghosts_aux)
    void exchange_ghosts(const DataArray& U, DataArray& Ughost) const;
private:
    Kokkos::View<uint32_t*> recv_sizes, send_sizes; //!Number of octants to send/recv for each proc
    Kokkos::View<uint32_t*>::HostMirror recv_sizes_host, send_sizes_host; //!Number of octants to send/recv for each proc
    Kokkos::View<uint32_t*> send_iOcts; //! List of octants to send (first send_sizes[0] iOcts to send to rank[0] and so on...)
    uint32_t nbghosts_recv;
    MpiComm mpi_comm;

public:
    /**
     * Generic function to exchange octant data stored un a Kokkos View
     * @tparam is the Kokkos::View type. It must be Kokkos::LayoutLeft
     * @tparam iOct_pos position of iOct coordinate in U. When iOct is not leftmost 
     *         coordinate (iOct_pos = DataArray_t::rank-1), packing/unpacking is less efficient 
     *         because data needs to be transposed.
     * @param U is local octant data with iOct_pos-nth subscript the octant index
     * @param Ughost is the ghost octant data to fill, it will be resized to match the number of ghost octants
     **/
    template< typename DataArray_t, int iOct_pos = DataArray_t::rank-1 >
    void exchange_ghosts_aux( const DataArray_t& U, DataArray_t& Ughost) const;
};

/**
 * Ghost communicator that does nothing.
 * To be used when MPI is disabled
 **/
class GhostCommunicator_serial: public GhostCommunicator_base
{
public:
    template< typename AMRmesh_t >
    GhostCommunicator_serial( std::shared_ptr<AMRmesh_t> amr_mesh )
    {}
    
    template< typename DataArray_t>
    void exchange_ghosts( const DataArray_t& U, DataArray_t& Ughost) const
    {
        assert(Ughost.size() == 0);
        /* Nothing to do */
    } 
};

#if DYABLO_USE_MPI
using GhostCommunicator = GhostCommunicator_kokkos;
#else
using GhostCommunicator = GhostCommunicator_serial;
#endif



}//namespace dyablo