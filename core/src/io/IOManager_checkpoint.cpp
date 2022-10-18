#include "io/IOManager_base.h"

#include "legacy/utils_block.h"
#include "legacy/io_utils.h"

#include "utils/monitoring/Timers.h"
#include "userdata_utils.h"

#include <hdf5.h>
#include <hdf5_hl.h>

namespace dyablo { 

template<typename T>
hid_t get_hdf5_type()
{
  static_assert( !std::is_same_v<T,T>, "get_hdf5_type not defined for this type" );
}
template<> hid_t get_hdf5_type<double>()    { return H5T_NATIVE_DOUBLE; }
template<> hid_t get_hdf5_type<uint32_t>()  { return H5T_NATIVE_UINT32; }
template<> hid_t get_hdf5_type<int32_t>()   { return H5T_NATIVE_INT32; }

class io_file{
private:
  hid_t m_hdf5_file;

public:
  io_file(const std::string& filename)
  {
    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
#ifdef DYABLO_USE_MPI
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
#endif
    m_hdf5_file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist);
    H5Pclose(plist);
  }

  // iOct should be outermost index
  template< typename T, int iOct_pos=T::rank-1 >
  void write_view( const std::string& name, const T& U )
  {
    hid_t hdf5_type = get_hdf5_type<typename T::value_type>();
    constexpr hid_t rank = T::rank;

    T U_right_iOct;
    userdata_utils::transpose_to_right_iOct<iOct_pos>( U, U_right_iOct );

    hsize_t local_dims[rank];
    hsize_t global_dims[rank];
    hsize_t local_start[rank];
    for( int i=1; i<rank; i++ )
    {
       local_dims[i] = U_right_iOct.extent(rank-1-i);
       global_dims[i] = U_right_iOct.extent(rank-1-i);
       local_start[i] = 0;
    }
    hid_t local_oct_count = U_right_iOct.extent(rank-1);
    hid_t global_oct_count;
    MPI_Allreduce( &local_oct_count, &global_oct_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD );
    // Perform exclusive prefix sum
    hid_t local_oct_start;
    MPI_Scan( &local_oct_count, &local_oct_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD );
    local_oct_start -= local_oct_count;
    local_dims[0] = local_oct_count;
    global_dims[0] = global_oct_count;
    local_start[0] = local_oct_start;

    hid_t filespace = H5Screate_simple( rank, global_dims, nullptr );
    hid_t  memspace = H5Screate_simple( rank,  local_dims, nullptr );

    // set some properties
    hid_t dataset_properties = H5Pcreate(H5P_DATASET_CREATE);
    hid_t write_properties   = H5Pcreate(H5P_DATASET_XFER);
    #ifdef DYABLO_USE_MPI
      H5Pset_dxpl_mpio(write_properties, H5FD_MPIO_COLLECTIVE);
    #endif

    hid_t dataset = H5Dcreate2( m_hdf5_file, name.c_str(), hdf5_type,
                                filespace, H5P_DEFAULT, dataset_properties, H5P_DEFAULT );
    H5Sselect_hyperslab( filespace, H5S_SELECT_SET, local_start, nullptr, local_dims, nullptr );
    H5Dwrite( dataset, hdf5_type, memspace, filespace, write_properties, U_right_iOct.data() );

    H5Dclose(dataset);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Pclose(dataset_properties);
    H5Pclose(write_properties);
  }

  ~io_file()
  {
    if (m_hdf5_file) {
      H5Fflush(m_hdf5_file, H5F_SCOPE_GLOBAL);
      H5Fclose(m_hdf5_file);
      m_hdf5_file = 0;
    }
  }
};

class ConfigMap_unchecked : public ConfigMap
{
public:
  ConfigMap_unchecked( const ConfigMap& configMap )
   : ConfigMap(configMap)
  {}

  template< typename T >
  void set(const std::string& section, const std::string& name, const T& val)
  {
    value_container& v = _values[section][name];
    v.value = Impl::to_string(val);
    v.from_file = false;
    v.used = true;
  }
};

class IOManager_checkpoint : public IOManager
{
public: 
  IOManager_checkpoint(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : pdata(new Data
      {
        configMap,
        foreach_cell.get_amr_mesh(), 
        timers,
        configMap.getValue<std::string>("output", "outputDir", "./"),
        configMap.getValue<std::string>("output", "outputPrefix", "output")
      })
  {}
  void save_snapshot( const ForeachCell::CellArray_global_ghosted& U, uint32_t iter, real_t time )
  {
    std::stringstream filename;
    filename << pdata->outputDir << "/restart_" << pdata->outputPrefix << "_" << iter;

    // Write HDF5 file
    {
      io_file hdf5_file(filename.str()+".h5");
      hdf5_file.write_view("U", U.U);
      // Select subview containing local octants
      auto oct_data = pdata->pmesh.getLightOctree().getStorage().getLocalSubview();
      // NOTE : write_view + subview only works when transposition is required
      hdf5_file.write_view<decltype(oct_data),0>("Octree", oct_data);
    }

    // Write .ini file
    {
      ConfigMap_unchecked configMap_copy = pdata->configMap;

      configMap_copy.set( "run", "tstart", time); 
      configMap_copy.set( "run", "iter_start", iter);
      configMap_copy.set( "hydro", "problem", "restart" );
      configMap_copy.set( "restart", "filename", filename.str()+".h5");

      std::ofstream ini_file(filename.str()+".ini");
      configMap_copy.output(ini_file);
    }
  }

  struct Data{ 
    const ConfigMap& configMap;
    AMRmesh& pmesh;
    Timers& timers;
    std::string outputDir, outputPrefix;
  };
private:
  std::unique_ptr<Data> pdata;
};

}// namespace dyablo


FACTORY_REGISTER( dyablo::IOManagerFactory, dyablo::IOManager_checkpoint, "IOManager_checkpoint" );

