#pragma once

#include "UserData.h"
#include "mpi/ViewCommunicator.h"

namespace dyablo {


class GhostCommunicator_full_blocks : protected ViewCommunicator
{
public:
    template< typename AMRmesh_t >
    GhostCommunicator_full_blocks( const AMRmesh_t& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape,  int ghost_count, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    : ViewCommunicator( ViewCommunicator::from_mesh(amr_mesh, mpi_comm) )
    {}

    static std::string name()
    {
      return "GhostCommunicator_full_blocks";
    }
     
    /// @copydoc GhostCommunicator_base::getNumGhosts
    uint32_t getNumGhosts() const
    {
      return ViewCommunicator::getNumGhosts();
    }

    void exchange_ghosts( const UserData::FieldAccessor& U ) const
    {
      for(int i=0; i<U.fm_ivar.nbfields(); i++)
      {
        int iVar = U.fm_active[i];
        auto U_subview      = Kokkos::subview( U.fields.U,      Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );
        auto Ughost_subview = Kokkos::subview( U.fields.Ughost, Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );

        ViewCommunicator::exchange_ghosts<2>(U_subview, Ughost_subview);
      }
    }

    void exchange_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
    {
      ViewCommunicator::exchange_ghosts<2>(U.U, U.Ughost);
    }

    void reduce_ghosts( UserData::FieldAccessor& U ) const
    {
      for(int i=0; i<U.fm_ivar.nbfields(); i++)
      {
        int iVar = U.fm_active[i];
        auto U_subview      = Kokkos::subview( U.fields.U,      Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );
        auto Ughost_subview = Kokkos::subview( U.fields.Ughost, Kokkos::ALL(), std::make_pair(iVar, iVar+1), Kokkos::ALL() );

        ViewCommunicator::reduce_ghosts<2>(U_subview, Ughost_subview);
      }
    }

    void reduce_ghosts( ForeachCell::CellArray_global_ghosted& U ) const
    {
      ViewCommunicator::reduce_ghosts<2>(U.U, U.Ughost);
    }  
};

} // namespace dyablo