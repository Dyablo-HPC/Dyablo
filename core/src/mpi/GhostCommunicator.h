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

    /// Get number of local ghosts
    //uint32_t getNumGhosts() const;

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

} // namespace dyablo

#include "GhostCommunicator_kokkos.h"

namespace dyablo {

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
    void exchange_ghosts(const DataArrayBlock& U, const DataArrayBlock& Ughost) const;
    
    /// Note : octant index for DataArray is leftmost subscript
    void exchange_ghosts(const DataArray& U, const DataArray& Ughost) const;
private:
    AMRmesh_pablo& amr_mesh;  
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
    void exchange_ghosts( const DataArray_t& U, const DataArray_t& Ughost) const
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