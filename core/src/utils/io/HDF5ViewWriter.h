#pragma once

#include <string>
#include "utils/mpi/GlobalMpiSession.h"

#include <hdf5.h>
#include <hdf5_hl.h>

namespace dyablo{

template< typename T >
hid_t hdf5_type_id() 
{
  static_assert( !std::is_same_v<T,T>, "Unknown type" );
  return 0;
}
template<> hid_t hdf5_type_id<float>   (){ return H5T_NATIVE_FLOAT; } 
template<> hid_t hdf5_type_id<double>  (){ return H5T_NATIVE_DOUBLE; } 
template<> hid_t hdf5_type_id<uint16_t>(){ return H5T_NATIVE_UINT16; } 
template<> hid_t hdf5_type_id<uint32_t>(){ return H5T_NATIVE_UINT32; } 
template<> hid_t hdf5_type_id<uint64_t>(){ return H5T_NATIVE_UINT64; }
template<> hid_t hdf5_type_id<int16_t> (){ return H5T_NATIVE_INT16; } 
template<> hid_t hdf5_type_id<int32_t> (){ return H5T_NATIVE_INT32; } 
template<> hid_t hdf5_type_id<int64_t> (){ return H5T_NATIVE_INT64; } 

class HDF5ViewWriter{

public:
  /**
   * Open a new hdf5 file at path `filename`
   **/
  HDF5ViewWriter( const std::string& filename, const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world() )
    : m_mpi_comm( mpi_comm )
  {
    hid_t file_params = H5Pcreate(H5P_FILE_ACCESS);
    #ifdef DYABLO_USE_MPI
      H5Pset_fapl_mpio(file_params, MPI_COMM_WORLD, MPI_INFO_NULL);
    #endif
    m_hdf5_file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, file_params);
  }

  ~HDF5ViewWriter()
  {
    close();
  }

  void close()
  {
    if (m_hdf5_file) {
      H5Fflush(m_hdf5_file, H5F_SCOPE_GLOBAL);
      H5Fclose(m_hdf5_file);
      m_hdf5_file = 0;
    }
  }

  /**
   * Same as `collective_write_hint` but `global_extent` and `first_global_index`
   * are determined using MPI communications.
   **/
  template< typename View_t >
  void collective_write( const std::string& varname, const View_t& data)
  {
    uint64_t local_extent = data.extent( View_t::rank-1 );
    uint64_t global_extent;
    m_mpi_comm.MPI_Allreduce( &local_extent, &global_extent, 1, MpiComm::MPI_Op_t::SUM );
    uint64_t first_global_index;
    m_mpi_comm.MPI_Scan( &local_extent, &first_global_index, 1, MpiComm::MPI_Op_t::SUM );
    first_global_index-=local_extent;

    collective_write_hint(varname, data, global_extent, first_global_index);
  }

  /**
   * Write collectively the content of the view `data` into the dataset names after `name`
   * @tparam a Kokkos::View with LayoutLeft
   * @param varname the name in the dataset inside the HDF5 file
   * @param data a Kokkos::View containing the data to write every extent should be identical on 
   *             every MPI process, except the last one. Data will be written contiguously for each MPI process.
   * @param global_extent cumulated values for each process for the last extent
   * @param first_global_index start value for the last extent of current process 
   **/
  template< typename View_t >
  void collective_write_hint( const std::string& varname, const View_t& data, 
                         uint64_t global_extent, uint64_t first_global_index )
  {
    static_assert( std::is_same_v< typename View_t::array_layout, Kokkos::LayoutLeft >, "View is not LayoutLeft" );

    // Read Rank and extent from view
    constexpr int view_rank = View_t::rank;
    hsize_t local_extents[view_rank];
    hsize_t global_extents[view_rank];
    hsize_t first_global_indexes[view_rank] = {};
    for( int i=0; i<view_rank; i++ )
    {
      local_extents[i] = data.extent(view_rank-1-i);
      global_extents[i] = data.extent(view_rank-1-i);
    }
    global_extents[0] = global_extent;
    first_global_indexes[0] = first_global_index;
    hid_t type_id = hdf5_type_id<typename View_t::value_type>();

    hid_t memspace;
    {
      memspace = H5Screate_simple(view_rank, local_extents, nullptr);
    }

    hid_t filespace;
    {
      filespace = H5Screate_simple(view_rank, global_extents, nullptr);
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, first_global_indexes, nullptr, local_extents, nullptr);
    }
    
    hid_t dataset;
    {
      hid_t dataset_properties = H5Pcreate(H5P_DATASET_CREATE);
      dataset = H5Dcreate2(m_hdf5_file, varname.c_str(), type_id, filespace, H5P_DEFAULT, dataset_properties, H5P_DEFAULT);
      H5Pclose(dataset_properties);
    }

    {
      hid_t write_properties = H5Pcreate(H5P_DATASET_XFER);
      #ifdef DYABLO_USE_MPI
        H5Pset_dxpl_mpio(write_properties, H5FD_MPIO_COLLECTIVE);
      #endif
      H5Dwrite(dataset, type_id, memspace, filespace, write_properties, data.data());
      H5Pclose(write_properties);
    }

    H5Dclose(dataset);
    H5Sclose(filespace);
    H5Sclose(memspace);
  }

private:
  hid_t m_hdf5_file; // HDF5 file descriptor
  MpiComm m_mpi_comm;
};

} // namespace dyablo