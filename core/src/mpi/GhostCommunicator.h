#pragma once

#include "amr/AMRmesh.h"

#include "GhostCommunicator_partial_blocks.h"
#include "GhostCommunicator_full_blocks.h"

namespace dyablo {

template< typename Impl >
class GhostCommunicator_impl : protected Impl
{
public:
  GhostCommunicator_impl( const AMRmesh& mesh, 
                          const ForeachCell::CellArray_global_ghosted::Shape_t& shape, 
                          int ghost_count, 
                          const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
  : Impl(mesh.getMesh(), shape, ghost_count, mpi_comm)
  {}

  uint32_t getNumGhosts() const
  {
    return Impl::getNumGhosts();
  }

  void exchange_ghosts( UserData::FieldAccessor& U ) const
  {
    Impl::exchange_ghosts(U);
  }

  void exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::exchange_ghosts(U);
  }

  void reduce_ghosts( UserData::FieldAccessor& U ) const
  {
    Impl::reduce_ghosts(U);
  }

  void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
  {
    Impl::reduce_ghosts(U);
  }

};

// GhostCommunicator full block implementation is only compatible with AMRmesh_hashmap_new
using GhostCommunicator = GhostCommunicator_impl< std::conditional_t< std::is_same_v<AMRmesh::Impl_t, AMRmesh_hashmap_new>, 
                                                                      GhostCommunicator_partial_blocks, 
                                                                      GhostCommunicator_full_blocks > >;

}// namespace dyablo
