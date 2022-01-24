#include "muscl_block/io/IOManager_hdf5.h"

#include "muscl_block/legacy/utils_block.h"
#include "shared/io_utils.h"
#include "shared/HDF5_IO.h"

#include "utils/monitoring/Timers.h"

namespace dyablo { 
namespace muscl_block {

struct IOManager_hdf5::Data{ 
  AMRmesh& pmesh;
  Timers& timers;  
  HDF5_Writer hdf5_writer;
  std::string outputDir, outputPrefix;
  std::string write_variables;
};

IOManager_hdf5::IOManager_hdf5(
  ConfigMap& configMap,
  ForeachCell& foreach_cell,
  Timers& timers )
 : pdata(new Data
    {foreach_cell.get_amr_mesh(), 
    timers,
    HDF5_Writer( &foreach_cell.get_amr_mesh(), configMap ),
    configMap.getValue<std::string>("output", "outputDir", "./"),
    configMap.getValue<std::string>("output", "outputPrefix", "output"),
    configMap.getValue<std::string>("output", "write_variables", "rho")
  })
{ 
}

IOManager_hdf5::~IOManager_hdf5()
{}

void IOManager_hdf5::save_snapshot( const ForeachCell::CellArray_global_ghosted& U_, uint32_t iter, real_t time )
{
  const FieldManager fieldMgr( U_.fm.enabled_fields() );
  const id2index_t& fm = U_.fm;
  HDF5_Writer* hdf5_writer = &(pdata->hdf5_writer);
  std::string write_variables = pdata->write_variables;

  DataArrayBlock U = U_.U;

  // a map containing ID and name of the variable to write
  str2int_t names2index; // this is initially empty
  build_var_to_write_map(names2index, fieldMgr, write_variables);

  // prepare output filename
  std::string outputPrefix = pdata->outputPrefix;
  std::string outputDir = pdata->outputDir;
  
  
  // prepare suffix string
  std::ostringstream strsuffix;
  strsuffix << "iter";
  strsuffix.width(7);
  strsuffix.fill('0');
  strsuffix << iter;

  // actual writing
  {
    DataArrayBlock::HostMirror Uhost = Kokkos::create_mirror_view(U);
    // copy device data to host
    Kokkos::deep_copy(Uhost, U);

    hdf5_writer->update_mesh_info();

    // open the new file and write our stuff
    std::string basename = outputPrefix + "_" + strsuffix.str();
    
    hdf5_writer->open(basename, outputDir);
    hdf5_writer->write_header(time);

    // write user the fake data (all scalar fields, here only one)
    hdf5_writer->write_quadrant_attribute(Uhost, fm, names2index);

    // check if we want to write velocity or rhoV vector fields
    // if (write_variables.find("velocity") != std::string::npos) {
    //   hdf5_writer->write_quadrant_velocity(U, fm, false);
    // } else if (write_variables.find("rhoV") != std::string::npos) {
    //   hdf5_writer->write_quadrant_velocity(U, fm, true);
    // } 
    
    if (write_variables.find("Mach") != std::string::npos) {
      // mach number will be recomputed from conservative variables
      // we could have used primitive variables, but since here Q
      // may not have the same size, Q may need to be resized
      // and recomputed anyway.
      hdf5_writer->write_quadrant_mach_number(Uhost, fm);
    }

    if (write_variables.find("P") != std::string::npos) {
      hdf5_writer->write_quadrant_pressure(Uhost, fm);
    }

    if (write_variables.find("iOct") != std::string::npos)
      hdf5_writer->write_quadrant_id(Uhost);

    // close the file
    hdf5_writer->write_footer();
    hdf5_writer->close();
  }
}

}// namespace dyablo
}// namespace muscl_block

FACTORY_REGISTER( dyablo::muscl_block::IOManagerFactory, dyablo::muscl_block::IOManager_hdf5, "IOManager_hdf5" );

