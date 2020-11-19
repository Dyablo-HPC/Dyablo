#pragma once

#include "shared/kokkos_shared.h"
#include "shared/bitpit_common.h"

namespace dyablo{
namespace muscl_block{

/**
 * Interface to implement a ghost communicator
 **/
class GhostCommunicator_base
{
public:
    using DataArray_t = DataArrayBlock;
    /**
     * Exchange ghost octants user data
     * @param U array containing user data (read only)
     * @param H_ghost array to contain ghost data
     **/
    void exchange_ghosts(DataArray_t& A, DataArray_t& A_ghost) const;
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
    void exchange_ghosts(DataArray_t& U, DataArray_t& Ughost) const;
private:
    std::shared_ptr<AMRmesh> amr_mesh;
};

using GhostCommunicator = GhostCommunicator_pablo;


}//namespace muscl_block
}//namespace dyablo