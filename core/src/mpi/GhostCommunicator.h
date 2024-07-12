#pragma once

#include "amr/AMRmesh.h"

#include "GhostCommunicator_partial_blocks.h"
#include "GhostCommunicator_full_blocks.h"

namespace dyablo {

template< typename Impl >
/***
 * Interface to implement for a GhostCommunicator
 ***/
class GhostCommunicator_impl : protected Impl
{
public:
  /**
   * @param mesh AMR mesh to determline neighborhood
   * @param shape shape of the blocks in the arrays
   * @param ghost_count number for ghosts needed for stencil operations (this is a minimum, actual comms can recieve more ghosts)
   * @param mpi_comm you know what it is
   **/
  GhostCommunicator_impl( const AMRmesh& mesh, 
                          const ForeachCell::CellArray_global_ghosted::Shape_t& shape, 
                          int ghost_count, 
                          const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
  : Impl(mesh.getMesh(), shape, ghost_count, mpi_comm)
  {}

  /// Number of ghost blocks (possibly partial blocks) for this mesh 
  uint32_t getNumGhosts() const
  {
    return Impl::getNumGhosts();
  }

  /***
   * Send ghosts cells for the selected Fields in the accessor
   * Depending on the backend, other fields from UserData could also be exchanged.
   * Cells at a distance greater than ghost_count from the local domain have undefined value
   * (they may be exchanged or not depending on the backend)
   ***/
  void exchange_ghosts( UserData::FieldAccessor& U ) const
  {
    Impl::exchange_ghosts(U);
  }

  /***
   * Send ghosts cells for all fields in the CellArray
   * Cells at a distance greater than ghost_count from the local domain have undefined value
   * (they may be exchanged or not depending on the backend)
   ***/
  void exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::exchange_ghosts(U);
  }


  /***
   * Reduce ghosts cells for the selected Fields in the accessor
   * BE SURE TO SET ALL YOUR GHOSTS TO ZERO TO AVOID ISSUES
   * 
   * Other fields from UserData WILL NOT be modified
   * Ghost Cells un neighboring blocks at a distance greater than ghost_count from 
   * the local domain may or may not be exchanged depending on the backend, be sure to set them to zero
   ***/
  void reduce_ghosts( UserData::FieldAccessor& U ) const
  {
    Impl::reduce_ghosts(U);
  }

  void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::reduce_ghosts(U);
  }

};

// GhostCommunicator partial block implementation is only compatible with AMRmesh_hashmap_new, use full block otherwise
using GhostCommunicator = GhostCommunicator_impl< std::conditional_t< std::is_same_v<AMRmesh::Impl_t, AMRmesh_hashmap_new>, 
                                                                      GhostCommunicator_partial_blocks, 
                                                                      GhostCommunicator_full_blocks > >;

}// namespace dyablo
