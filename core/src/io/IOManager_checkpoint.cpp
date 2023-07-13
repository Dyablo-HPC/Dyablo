#include "io/IOManager_base.h"

#include "legacy/utils_block.h"
#include "legacy/io_utils.h"

#include "utils/monitoring/Timers.h"
#include "userdata_utils.h"

#include "utils/io/HDF5ViewWriter.h"
#include "userdata_utils.h"

namespace dyablo { 

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
  void save_snapshot( const UserData& U, ScalarSimulationData& scalar_data )
  {
    int iter = scalar_data.get<int>( "iter" );
    real_t time = scalar_data.get<real_t>( "time" );

    std::stringstream filename;
    filename << pdata->outputDir << "/restart_" << pdata->outputPrefix << "_" << iter;

    // Write HDF5 file
    {
      HDF5ViewWriter hdf5_file(filename.str()+".h5");
      
      for( std::string field_name : U.getEnabledFields() )
      {
        // TODO save collective_write hint to avoid allreduce for sizes each time
        hdf5_file.collective_write( std::string("fields/")+field_name, U.getField(field_name).U );
      }

      for( const std::string& particle_array : U.getEnabledParticleArrays() )
      {
        const auto& pos = U.getParticleArray(particle_array).particle_position;
        hdf5_file.collective_write( std::string("particles/")+particle_array+"/pos/x", Kokkos::subview( pos, Kokkos::ALL(), (int)IX ) );
        hdf5_file.collective_write( std::string("particles/")+particle_array+"/pos/y", Kokkos::subview( pos, Kokkos::ALL(), (int)IY ) );
        hdf5_file.collective_write( std::string("particles/")+particle_array+"/pos/z", Kokkos::subview( pos, Kokkos::ALL(), (int)IZ ) );
        for( const std::string& attr : U.getEnabledParticleAttributes( particle_array ) )
          hdf5_file.collective_write( std::string("particles/")+particle_array+"/attributes/"+attr, U.getParticleAttribute(particle_array, attr).particle_data );
      }

      // Select subview containing local octants
      LightOctree::Storage_t::oct_data_t oct_data = pdata->pmesh.getLightOctree().getStorage().getLocalSubview();
      LightOctree::Storage_t::oct_data_t oct_data_transpose;
      userdata_utils::transpose_to_right_iOct<0>( oct_data, oct_data_transpose );
      hdf5_file.collective_write("Octree", oct_data_transpose);

      hdf5_file.write_scalar("test/test", GlobalMpiSession::get_comm_world().MPI_Comm_rank()+1);

      scalar_data.foreach_var( [&]( const std::string& name, auto val )
      {
        hdf5_file.write_scalar(std::string("scalar_data/")+name, val);
      });
    }

    // Write .ini file
    {
      ConfigMap_unchecked configMap_copy = pdata->configMap;

      configMap_copy.set( "run", "tstart", time); 
      configMap_copy.set( "run", "iter_start", iter);
      configMap_copy.set( "run", "initial_conditions", "restart" );
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

