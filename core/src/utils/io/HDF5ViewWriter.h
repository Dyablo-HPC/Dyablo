#pragma once

#include <string>
#include "utils/mpi/GlobalMpiSession.h"

#include <hdf5.h>
#include <hdf5_hl.h>

namespace dyablo{

template< typename T >
inline hid_t hdf5_type_id() 
{
  static_assert( !std::is_same_v<T,T>, "Unknown type" );
  return 0;
}
template<> inline hid_t hdf5_type_id<float>   (){ return H5T_NATIVE_FLOAT; } 
template<> inline hid_t hdf5_type_id<double>  (){ return H5T_NATIVE_DOUBLE; } 
template<> inline hid_t hdf5_type_id<uint16_t>(){ return H5T_NATIVE_UINT16; } 
template<> inline hid_t hdf5_type_id<uint32_t>(){ return H5T_NATIVE_UINT32; } 
template<> inline hid_t hdf5_type_id<uint64_t>(){ return H5T_NATIVE_UINT64; }
template<> inline hid_t hdf5_type_id<int16_t> (){ return H5T_NATIVE_INT16; } 
template<> inline hid_t hdf5_type_id<int32_t> (){ return H5T_NATIVE_INT32; } 
template<> inline hid_t hdf5_type_id<int64_t> (){ return H5T_NATIVE_INT64; } 

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

  template< typename T >
  void write_scalar( const std::string& varpath, const T& value)
  {
    hid_t type_id = hdf5_type_id<T>();
    hid_t filespace = H5Screate(H5S_SCALAR);
    hid_t dataset = H5Pcreate(H5P_ATTRIBUTE_CREATE);

    hid_t group_id;
    std::string varname;
    {
      auto slash_pos = varpath.find_last_of('/');
      
      std::string group_path;
      if( slash_pos != std::string::npos )
      {
        group_path = varpath.substr( 0, slash_pos );
        varname = varpath.substr( slash_pos+1 );
      }
      else
      {
        group_path = "";
        varname = varpath;
      }

      group_id = get_group(group_path);
    }

    hid_t attr = H5Acreate2(group_id, varname.c_str(), type_id, filespace, dataset, H5P_DEFAULT);

    H5Awrite( attr, type_id, &value );
  
    H5Aclose(attr);
    H5Gclose(group_id);
    H5Pclose(dataset);
    H5Sclose(filespace);
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
   * @param varpath the name in the dataset inside the HDF5 file
   * @param data a Kokkos::View containing the data to write every extent should be identical on 
   *             every MPI process, except the last one. Data will be written contiguously for each MPI process.
   * @param global_extent cumulated values for each process for the last extent
   * @param first_global_index start value for the last extent of current process 
   **/
  template< typename View_t >
  void collective_write_hint( const std::string& varpath, const View_t& data, 
                         uint64_t global_extent, uint64_t first_global_index )
  {
    static_assert( std::is_same_v< typename View_t::array_layout, Kokkos::LayoutLeft >, "View is not LayoutLeft" );

    // Read Rank and extent from view
    int view_rank = View_t::rank;
    std::vector<hsize_t> local_extents(view_rank);
    std::vector<hsize_t> global_extents(view_rank);
    std::vector<hsize_t> first_global_indexes(view_rank);
    for( int i=0; i<view_rank; i++ )
    {
      first_global_indexes[i] = 0;
      local_extents[i] = data.extent(view_rank-1-i);
      global_extents[i] = data.extent(view_rank-1-i);
    }
    global_extents[0] = global_extent;
    first_global_indexes[0] = first_global_index;
    hid_t type_id = hdf5_type_id<typename View_t::value_type>();

    hid_t memspace;
    {
      memspace = H5Screate_simple(view_rank, local_extents.data(), nullptr);
    }

    hid_t filespace;
    {
      filespace = H5Screate_simple(view_rank, global_extents.data(), nullptr);
      H5Sselect_hyperslab(filespace, H5S_SELECT_SET, first_global_indexes.data(), nullptr, local_extents.data(), nullptr);
    }
    
    hid_t group_id;
    std::string varname;
    {
      auto slash_pos = varpath.find_last_of('/');
      
      std::string group_path;
      if( slash_pos != std::string::npos )
      {
        group_path = varpath.substr( 0, slash_pos );
        varname = varpath.substr( slash_pos+1 );
      }
      else
      {
        group_path = "";
        varname = varpath;
      }

      group_id = get_group(group_path);
    }

    hid_t dataset;
    {
      hid_t dataset_properties = H5Pcreate(H5P_DATASET_CREATE);
      dataset = H5Dcreate2(group_id, varname.c_str(), type_id, filespace, H5P_DEFAULT, dataset_properties, H5P_DEFAULT);
      H5Pclose(dataset_properties);
    }

    {
      hid_t write_properties = H5Pcreate(H5P_DATASET_XFER);
      #ifdef DYABLO_USE_MPI
        H5Pset_dxpl_mpio(write_properties, H5FD_MPIO_COLLECTIVE);
      #endif
      #ifdef HDF5_IS_CUDA_AWARE
      {
        Kokkos::fence();
        H5Dwrite(dataset, type_id, memspace, filespace, write_properties, data.data());
      }
      #else
      {
        auto data_host = Kokkos::create_mirror_view( data );
        Kokkos::deep_copy( data_host, data );
        H5Dwrite(dataset, type_id, memspace, filespace, write_properties, data_host.data());
      }
      #endif
      H5Pclose(write_properties);
    }

    H5Dclose(dataset);
    if( group_id != m_hdf5_file )
      H5Gclose(group_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
  }

private:
  hid_t m_hdf5_file; // HDF5 file descriptor
  MpiComm m_mpi_comm;

  hid_t get_group(const std::string& group_path)
  {
    std::stringstream ss (group_path);
    hid_t group_id = m_hdf5_file;
    std::string group_name;
    while( getline (ss, group_name, '/' ) )
    {
      hid_t group_id_old = group_id;
      htri_t group_exists = H5Lexists(group_id, group_name.c_str(), H5P_DEFAULT);
      if( group_exists <= 0 ) // Does not exist : create
        group_id = H5Gcreate(group_id, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      else // Group exists
        group_id = H5Gopen(group_id, group_name.c_str(), H5P_DEFAULT);
      
      if( group_id_old != m_hdf5_file )
        H5Gclose(group_id_old);
    }
    return group_id;
  }
};

} // namespace dyablo
