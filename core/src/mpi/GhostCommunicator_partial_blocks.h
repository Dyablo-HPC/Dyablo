#pragma once

#include <vector>
#include "Kokkos_Core.hpp"
#include "foreach_cell/ForeachCell.h"

namespace dyablo {

/**
 * Ghost communicator for partial blocks (ghosts) 
 * then serialize/deserialize in Kokkos kernels and use CUDA-aware MPI 
 **/
class GhostCommunicator_partial_blocks
{
public:
    GhostCommunicator_partial_blocks( const AMRmesh_hashmap_new& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape,  int ghost_count, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() );

    void init( const AMRmesh_hashmap_new& amr_mesh, const ForeachCell::CellArray_global_ghosted::Shape_t& shape, int ghost_count, const MpiComm& mpi_comm );
    
    /// @copydoc GhostCommunicator_base::getNumGhosts
    uint32_t getNumGhosts() const
    {
      return this->m_local_ghost_octants;
    }

    /**
     * TODO : doc
     **/
    template< typename CellArray_t >
    void exchange_ghosts( CellArray_t& U) const
    {
      using CellIndex = ForeachCell::CellIndex;

      uint32_t num_vars = U.nbFields(); // number of vars for each cell

      // Number of values to send are cell_count * num_vars 
      std::vector<int> send_sizes = this->m_send_cell_count;
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_recv_cell_count;  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const Kokkos::View< uint32_t* >& send_iOct = this->m_send_iOct;
      const Kokkos::View< uint32_t* >& send_iCell = this->m_send_iCell;
      const Kokkos::View< uint32_t* >& recv_iOct = this->m_recv_iOct;
      const Kokkos::View< uint32_t* >& recv_iCell = this->m_recv_iCell;
      uint32_t total_send_size = send_iOct.size(), total_recv_size = recv_iOct.size(); // send/recv buffer size (number of cells)    
      uint32_t bx=U.getShape().bx, by=U.getShape().by, bz=U.getShape().bz ; // Block size

      Kokkos::View< real_t*, Kokkos::LayoutLeft > send_buffer("exchange_ghosts::send_buffer", num_vars*total_send_size );

      Kokkos::parallel_for("exchange_ghosts::pack", total_send_size*num_vars,
        KOKKOS_LAMBDA( uint32_t ipack )
      {
        uint32_t ighost = ipack/num_vars;
        uint32_t ivar = ipack%num_vars;

        uint32_t iOct = send_iOct(ighost);
        uint32_t iCell = send_iCell(ighost);
        uint32_t i = iCell%bx; 
        uint32_t j = (iCell/bx)%by; 
        uint32_t k = (iCell/bx)/by;

        CellIndex cell_index { {iOct, false}, i, j, k, bx, by, bz };
        send_buffer( ipack ) = U.at_ivar( cell_index, ivar );
      });


      Kokkos::View< real_t* > recv_buffer("exchange_ghosts::recv_buffer", num_vars*total_recv_size ); 
    #ifdef MPI_IS_CUDA_AWARE 
      Kokkos::fence();
      mpi_comm.MPI_Alltoallv( send_buffer.data(), send_sizes.data(), recv_buffer.data(), recv_sizes.data() );
      Kokkos::fence();
    #else
      {
        auto send_buffer_host = Kokkos::create_mirror_view(send_buffer);
        auto recv_buffer_host = Kokkos::create_mirror_view(recv_buffer);

        Kokkos::deep_copy(send_buffer_host, send_buffer);
        mpi_comm.MPI_Alltoallv( send_buffer_host.data(), send_sizes.data(), recv_buffer_host.data(), recv_sizes.data() );
        Kokkos::deep_copy(recv_buffer, recv_buffer_host);
      }  
    #endif

      Kokkos::parallel_for("exchange_ghosts::unpack", total_recv_size*num_vars,
        KOKKOS_LAMBDA( uint32_t ipack )
      {
        uint32_t ighost = ipack/num_vars;
        uint32_t ivar = ipack%num_vars;

        uint32_t iOct = recv_iOct(ighost);
        uint32_t iCell = recv_iCell(ighost);
        uint32_t i = iCell%bx; 
        uint32_t j = (iCell/bx)%by; 
        uint32_t k = (iCell/bx)/by;

        CellIndex cell_index { {iOct, true}, i, j, k, bx, by, bz };
        U.at_ivar( cell_index, ivar ) = recv_buffer( ipack );
      });
    }

    /**
     * TODO : doc
     **/
    template< typename CellArray_t >
    void reduce_ghosts( CellArray_t& U) const
    {
      
      using CellIndex = ForeachCell::CellIndex;

      uint32_t num_vars = U.nbFields(); // number of vars for each cell

      // Note : sends and recvs counts are swapped for reduce

      // Number of values to send are cell_count * num_vars 
      std::vector<int> send_sizes = this->m_recv_cell_count;
      for( auto& v : send_sizes )
        v*=num_vars;
      std::vector<int> recv_sizes = this->m_send_cell_count;  
      for( auto& v : recv_sizes )
        v*=num_vars;

      const Kokkos::View< uint32_t* >& send_iOct = this->m_recv_iOct;
      const Kokkos::View< uint32_t* >& send_iCell = this->m_recv_iCell;
      const Kokkos::View< uint32_t* >& recv_iOct = this->m_send_iOct;
      const Kokkos::View< uint32_t* >& recv_iCell = this->m_send_iCell;
      uint32_t total_send_size = send_iOct.size(), total_recv_size = recv_iOct.size(); // send/recv buffer size (number of cells)    
      uint32_t bx=U.getShape().bx, by=U.getShape().by, bz=U.getShape().bz ; // Block size
      
      Kokkos::View< real_t*, Kokkos::LayoutLeft > send_buffer("reduce_ghosts::send_buffer", num_vars*total_send_size );
    
      Kokkos::parallel_for("reduce_ghosts::pack", total_send_size*num_vars,
        KOKKOS_LAMBDA( uint32_t ipack )
      {
        uint32_t ighost = ipack/num_vars;
        uint32_t ivar = ipack%num_vars;

        uint32_t iOct = send_iOct(ighost);
        uint32_t iCell = send_iCell(ighost);
        uint32_t i = iCell%bx; 
        uint32_t j = (iCell/bx)%by; 
        uint32_t k = (iCell/bx)/by;

        CellIndex cell_index { {iOct, true}, i, j, k, bx, by, bz };
        send_buffer( ipack ) = U.at_ivar( cell_index, ivar );
      });
      
      Kokkos::View< real_t* > recv_buffer("exchange_ghosts::recv_buffer", num_vars*total_recv_size ); 
    #ifdef MPI_IS_CUDA_AWARE 
      Kokkos::fence();
      mpi_comm.MPI_Alltoallv( send_buffer.data(), send_sizes.data(), recv_buffer.data(), recv_sizes.data() );
      Kokkos::fence();
    #else
      {
        auto send_buffer_host = Kokkos::create_mirror_view(send_buffer);
        auto recv_buffer_host = Kokkos::create_mirror_view(recv_buffer);

        Kokkos::deep_copy(send_buffer_host, send_buffer);
        mpi_comm.MPI_Alltoallv( send_buffer_host.data(), send_sizes.data(), recv_buffer_host.data(), recv_sizes.data() );
        Kokkos::deep_copy(recv_buffer, recv_buffer_host);
      }  
    #endif

      Kokkos::parallel_for("reduce_ghosts::unpack", total_recv_size*num_vars,
        KOKKOS_LAMBDA( uint32_t ipack )
      {
        uint32_t ighost = ipack/num_vars;
        uint32_t ivar = ipack%num_vars;

        uint32_t iOct = recv_iOct(ighost);
        uint32_t iCell = recv_iCell(ighost);
        uint32_t i = iCell%bx; 
        uint32_t j = (iCell/bx)%by; 
        uint32_t k = (iCell/bx)/by;

        CellIndex cell_index { {iOct, false}, i, j, k, bx, by, bz };
        Kokkos::atomic_add( &U.at_ivar( cell_index, ivar ), recv_buffer( ipack ) );
      });
    }

private:
    uint32_t m_local_ghost_octants; // Number of octants to allocate for ghosts
    std::vector<int> m_send_cell_count; // number of cells to send to each process
    std::vector<int> m_recv_cell_count; // number of cells to recv from each process
    Kokkos::View< uint32_t* > m_send_iOct; // send_iOct(ighost) iOct of cell to pack to position ighost in send buffer
    Kokkos::View< uint32_t* > m_send_iCell; // send_iCell(ighost) iCell of cell to pack to position ighost in send buffer
    Kokkos::View< uint32_t* > m_recv_iOct; // recv_iOct(ighost) iOct of cell to unpack from position ighost in recv buffer
    Kokkos::View< uint32_t* > m_recv_iCell; // recv_iCell(ighost) iCell of cell to unpack from position ighost in recv buffer
    
    MpiComm mpi_comm;    
};

} // namespace dyablo