#include "HDF5_IO.h"

#include <stdio.h>


#define IO_NODES_PER_CELL_2D 4
#define IO_TOPOLOGY_TYPE_2D "Quadrilateral"
#define IO_NODES_PER_CELL_3D 8
#define IO_TOPOLOGY_TYPE_3D "Hexahedron"

#define IO_XDMF_NUMBER_TYPE "NumberType=\"Float\" Precision=\"4\""

#ifndef IO_MPIRANK_WRAP
#define IO_MPIRANK_WRAP 128
#endif

#ifndef IO_HDF5_COMPRESSION
#define IO_HDF5_COMPRESSION 3
#endif

namespace dyablo {

/**
 * \brief Convert a H5T_NATIVE_* type to an XDMF NumberType
 */
static const char  *
hdf5_native_type_to_string (hid_t type)
{
  
  if (type == H5T_NATIVE_INT || type == H5T_NATIVE_LONG) {
    return "NumberType=\"Int\"";
  }

  if (type == H5T_NATIVE_UINT || type == H5T_NATIVE_ULONG) {
    return "NumberType=\"UInt\"";
  }

  if (type == H5T_NATIVE_CHAR) {
    return "NumberType=\"Char\"";
  }

  if (type == H5T_NATIVE_UCHAR) {
    return "NumberType=\"UChar\"";
  }

  if (type == H5T_NATIVE_FLOAT || type == H5T_NATIVE_DOUBLE) {
    return IO_XDMF_NUMBER_TYPE;
  }

  // INFOF("Unsupported number type %ld.\n", (long int)type);
  return IO_XDMF_NUMBER_TYPE;

} // hdf5_native_type_to_string

// =======================================================
// =======================================================
HDF5_Writer::HDF5_Writer(std::shared_ptr<AMRmesh> amr_mesh, 
                         ConfigMap& configMap,
                         HydroParams& params) :
  m_amr_mesh(amr_mesh),
  m_configMap(configMap),
  m_params(params)
{

  m_write_mesh_info = m_configMap.getBool("output", "write_mesh_info", false);

  // only meaningful when one wants to write block data
  m_write_block_data = m_configMap.getBool("amr", "use_block_data", false);
  m_bx = m_configMap.getInteger("amr", "bx", 0);
  m_by = m_configMap.getInteger("amr", "by", 0);
  m_bz = m_configMap.getInteger("amr", "bz", 0);

  m_nbCellsPerLeaf = 1;

  if (m_write_block_data) {
    m_nbCellsPerLeaf = m_params.dimType == TWO_D ? 
      m_bx * m_by : 
      m_bx * m_by * m_bz;
  }

  m_write_level = m_write_mesh_info;
  m_write_rank =  m_write_mesh_info;

  m_write_iOct = m_configMap.getBool("output", "write_iOct", false);

  m_nbNodesPerCell = m_params.dimType==TWO_D ? 
    IO_NODES_PER_CELL_2D : 
    IO_NODES_PER_CELL_3D;
  
  update_mesh_info();

  //printf("%d %d %d %d\n",m_local_num_quads,m_global_num_quads,m_local_num_nodes,m_global_num_nodes);

  m_basename = ""; // NOT VERY clean: setup in open
  m_hdf5_file = 0;
  m_xdmf_file = nullptr;

  m_mpiRank = m_amr_mesh->getRank();

  // is actually hdf5 enabled ?
  bool hdf5_enabled = m_configMap.getBool("output", "hdf5_enabled", false);

  if (m_mpiRank == 0 and hdf5_enabled) {

    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
    std::string filename;
    filename = outputPrefix + "_main.xmf";
    
    // INFOF("Writing main XMDF file \"%s\".\n", filename.c_str());
    m_main_xdmf_file = fopen(filename.c_str(), "w");
    io_xdmf_write_main_header();
  } else {
    m_main_xdmf_file = nullptr;
  }

} // HDF5_Writer::HDF5_Writer

// =======================================================
// =======================================================
HDF5_Writer::~HDF5_Writer()
{
  
  /*
   * Only rank 0 needs to close the main xdmf file.
   */
  if (m_mpiRank == 0 and m_main_xdmf_file != nullptr) {
    io_xdmf_write_main_footer();

    fflush(m_main_xdmf_file);
    fclose(m_main_xdmf_file);
    m_main_xdmf_file = nullptr;
  }

  // close other file
  close();
  
} // HDF5_Writer::~HDF5_Writer

// =======================================================
// =======================================================
void
HDF5_Writer::update_mesh_info()
{

  m_local_num_quads = m_amr_mesh->getNumOctants();
  m_global_num_quads = m_amr_mesh->getGlobalNumOctants();

  m_local_num_nodes = m_nbNodesPerCell * m_local_num_quads;
  m_global_num_nodes = m_nbNodesPerCell * m_global_num_quads;

} // HDF5_Writer::update_mesh_info

// =======================================================
// =======================================================
void
HDF5_Writer::open(std::string basename)
{
  hid_t               plist;
  std::string         filename;

  /*
   * Open parallel HDF5 resources.
   */
  plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist, m_amr_mesh->getComm(), MPI_INFO_NULL);

  filename = basename + ".h5";
  m_basename = basename;
  m_hdf5_file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist);
  H5Pclose(plist);

  // open xdmf files (one for each hdf5, a main xdmf file)
  if (m_amr_mesh->getRank() == 0) {

    filename = basename + ".xmf";
    m_xdmf_file = fopen(filename.c_str(), "w");

    if (m_main_xdmf_file) {
      io_xdmf_write_main_include(filename);
    }

  }
  
} // HDF5_Writer::open

// =======================================================
// =======================================================
void
HDF5_Writer::close()
{
  // close HDF5 file descriptor
  if (m_hdf5_file) {
    H5Fflush(m_hdf5_file, H5F_SCOPE_GLOBAL);
    H5Fclose(m_hdf5_file);
    m_hdf5_file = 0;
  }

  // close XDMF file descriptor
  if (m_xdmf_file) {
    fflush(m_xdmf_file);
    fclose(m_xdmf_file);
    m_xdmf_file = nullptr;
  }
  
} // HDF5_Writer::close

// =======================================================
// =======================================================
int
HDF5_Writer::write_header(double time)
{

  // write the xmdf file first
  if (m_mpiRank == 0) {
    io_xdmf_write_header(time);
  }

  // and write stuff into the hdf file
  io_hdf5_write_coordinates();
  io_hdf5_write_connectivity();
  io_hdf5_write_level();
  io_hdf5_write_rank();
  io_hdf5_write_iOct();

  return 0;

} // HDF5_Writer::write_header

// =======================================================
// =======================================================
int
HDF5_Writer::write_footer()
{
  int mpirank = m_amr_mesh->getRank();

  if (mpirank == 0) {
    io_xdmf_write_footer();
  }

  return 0;
  
} // HDF5_Writer::write_footer

// =======================================================
// =======================================================
int
HDF5_Writer::write_attribute(const std::string &name,
                             void *data,
                             size_t dim,
                             io_attribute_type_t ftype,
                             hid_t dtype,
                             hid_t wtype)
{

  hsize_t             dims[2] = { 0, 0 };
  hsize_t             count[2] = { 0, 0 };
  hsize_t             start[2] = { 0, 0 };
  int                 rank = 0;

  if (ftype == IO_CELL_SCALAR || ftype == IO_CELL_VECTOR) {

    dims[0] = m_amr_mesh->getGlobalNumOctants()*m_nbCellsPerLeaf;
    dims[1] = dim;

    count[0] = m_amr_mesh->getNumOctants()*m_nbCellsPerLeaf;
    count[1] = dims[1];

    // get global index of the first octant of current mpi processor
    start[0] = m_amr_mesh->getGlobalIdx((uint32_t) 0)*m_nbCellsPerLeaf;
    start[1] = 0;

  } else {

    // is this relevant ?

    // dims[0] = m_global_num_nodes;
    // dims[1] = dim;

    // count[0] = m_local_num_nodes;
    // count[1] = dims[1];

    // start[0] = m_start_nodes;
    // start[1] = 0;

  }

  if (ftype == IO_CELL_SCALAR || ftype == IO_NODE_SCALAR) {
    rank = 1;
  } else {
    rank = 2;
  }

  // TODO: find a better way to pass the number type
  if (m_mpiRank == 0) {
    const char *dtype_str = hdf5_native_type_to_string(dtype);
    io_xdmf_write_attribute(name, dtype_str, ftype, dims);
  }

  io_hdf5_writev(m_hdf5_file, name, data, dtype, wtype, rank, dims, count, start);

  return 0;
  
} // HDF5_Writer::write_attribute

// =======================================================
// =======================================================
int
HDF5_Writer::write_quadrant_attribute(DataArray  data,
                                      id2index_t fm,
                                      str2int_t  names2index)
{

  // copy data from device to host
  DataArrayHost datah = Kokkos::create_mirror(data);
  // copy device data to host
  Kokkos::deep_copy(datah, data);
  
  // write data array scalar fields in ascii
  for ( auto iter : names2index) {
    
    // get variables string name
    const std::string varName = iter.first;
    
    // get variable id
    int iVar = iter.second;

    // if DataArray has a left layout, we only need to define
    // a slice to actual scalar data
    // if DataArray has right layout, we need to actually extract
    // the slide so that it is memory contiguous
    if ( std::is_same< 
         DataArray::array_layout,
         Kokkos::LayoutLeft >::value) {

      auto dataVar = Kokkos::subview(data, Kokkos::ALL(), fm[iVar]);

      // actual data writing
      write_attribute(varName, dataVar.ptr_on_device(),
                      0, IO_CELL_SCALAR,
                      H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE);
    } else {

      using DataArrayScalar = Kokkos::View<real_t*, Kokkos::HostSpace>;
      
      DataArrayScalar dataVar = DataArrayScalar("scalar_array_for_hdf5_io",data.extent(0));

      uint32_t nbOcts = data.extent(0);

      Kokkos::parallel_for(nbOcts, KOKKOS_LAMBDA (uint32_t iOct) {
          dataVar(iOct) = data(iOct,fm[iVar]);
        });

      // actual data writing
      write_attribute(varName, dataVar.ptr_on_device(), 
                      0, IO_CELL_SCALAR,
                      H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE);

    }

  } // end for iter
    
  return 0;

} // HDF5_Writer::write_quadrant_attribute

// =======================================================
// =======================================================
int
HDF5_Writer::write_quadrant_velocity(DataArray  data,
                                     id2index_t fm,
                                     bool use_momentum)
{

  using DataArrayVector = Kokkos::View<real_t*, Kokkos::HostSpace>;    

  // copy data from device to host
  DataArrayHost datah = Kokkos::create_mirror(data);
  // copy device data to host
  Kokkos::deep_copy(datah, data);
  
  int dim = fm[IW]==-1 ? 2 : 3;

  uint32_t nbOcts = data.extent(0);

  DataArrayVector dataVector = DataArrayVector("temp_array_hdf5", nbOcts*3);

  Kokkos::parallel_for(
    nbOcts, KOKKOS_LAMBDA(uint32_t iOct) {
        if (use_momentum) {
          dataVector(3 * iOct + 0) = data(iOct, fm[IU]);
          dataVector(3 * iOct + 1) = data(iOct, fm[IV]);
          //if (dim == 3)
          dataVector(3 * iOct + 2) = dim==3 ? data(iOct, fm[IW]) : 0;
        } else {
          dataVector(3 * iOct + 0) = data(iOct, fm[IU])/data(iOct, fm[ID]);
          dataVector(3 * iOct + 1) = data(iOct, fm[IV])/data(iOct, fm[ID]);
          //if (3 == 3)
          dataVector(3 * iOct + 2) = dim==3 ? data(iOct, fm[IW])/data(iOct, fm[ID]) : 0;
        }
      });

  // actual data writing
  const std::string varName = use_momentum ? "rhoV" : "velocity";
  write_attribute(varName, dataVector.ptr_on_device(),
                  3 /*dim*/, IO_CELL_VECTOR,
                  H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE);
  
  return 0;

} // HDF5_Writer::write_quadrant_velocity

// =======================================================
// =======================================================
int
HDF5_Writer::write_quadrant_mach_number(DataArray  data,
                                        id2index_t fm)
{
  // copy data from device to host
  DataArrayHost datah = Kokkos::create_mirror(data);
  
  // copy device data to host
  Kokkos::deep_copy(datah, data);
  
  {
    
    uint32_t nbOcts = data.extent(0);

    using DataArrayScalar = Kokkos::View<real_t*, Kokkos::HostSpace>;
    
    DataArrayScalar mach_number = DataArrayScalar("mach_number", nbOcts);
    
    Kokkos::parallel_for(nbOcts, KOKKOS_LAMBDA (uint32_t iOct) {
        
        real_t d = datah(iOct,fm[ID]);
        real_t u = datah(iOct,fm[IU])/datah(iOct,fm[ID]);
        real_t v = datah(iOct,fm[IV])/datah(iOct,fm[ID]);
        real_t w = fm[IW]==-1 ? 0 : datah(iOct,fm[IW])/datah(iOct,fm[ID]);

        // kinetic energy
        real_t eken = 0.5*d*(u*u+v*v+w*w); 

        // internal energy
        real_t eint = datah(iOct,fm[IE])-eken;

        // specific heat ratio
        real_t gamma0 = m_params.settings.gamma0;

        // pressure
        real_t p = (gamma0-1)*eint;

        // square speed of sound
        real_t cs2 = p/d;

        // velocity square
        real_t u2 = u*u+v*v+w*w;

        mach_number(iOct) = sqrt(u2/cs2);

      });

    // actual data writing
    write_attribute("Mach", mach_number.ptr_on_device(), 
                    0, IO_CELL_SCALAR,
                    H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE);

  }
    
  return 0;

} // HDF5_Writer::write_quadrant_mach_number

// =======================================================
// =======================================================
int
HDF5_Writer::write_quadrant_attribute(DataArrayBlock  data,
                                      id2index_t      fm,
                                      str2int_t       names2index)
{

  // copy data from device to host
  DataArrayBlockHost datah = Kokkos::create_mirror(data);
  
  // copy device data to host
  Kokkos::deep_copy(datah, data);
  
  // write data array scalar fields in ascii
  for ( auto iter : names2index) {
    
    // get variables string name
    const std::string varName = iter.first;
    
    // get variable id
    int iVar = iter.second;

    uint32_t nbCellsPerOct = data.extent(0);
    uint32_t nbOcts = data.extent(2);

    // we need to gather data corresponding to a given scalar variable
    using DataArrayScalar = Kokkos::View<real_t*, Kokkos::HostSpace>;

    // remember that
    // - data.extent(0) is the number of cells per octant
    // - data.extent(1) is the number of scalar fields
    // - data.extent(2) is the total number of oct in current MPI process
    DataArrayScalar dataVar = DataArrayScalar("scalar_array_for_hdf5_io", nbCellsPerOct*nbOcts);

    Kokkos::parallel_for(nbOcts, KOKKOS_LAMBDA (uint32_t iOct) {
        for (uint32_t iCell=0; iCell<nbCellsPerOct; ++iCell)
          dataVar(iCell + nbCellsPerOct*iOct) = data(iCell,fm[iVar],iOct);
      });
    
    // actual data writing
    write_attribute(varName, dataVar.ptr_on_device(), 
                    0, IO_CELL_SCALAR,
                    H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE);

  } // end for iter
    
  return 0;

} // HDF5_Writer::write_quadrant_attribute

// =======================================================
// =======================================================
int
HDF5_Writer::write_quadrant_mach_number(DataArrayBlock data,
                                        id2index_t fm)
{
  // copy data from device to host
  DataArrayBlockHost datah = Kokkos::create_mirror(data);
  
  // copy device data to host
  Kokkos::deep_copy(datah, data);
  
  {
    
    using DataArrayScalar = Kokkos::View<real_t*, Kokkos::HostSpace>;

    uint32_t nbCellsPerOct = data.extent(0);
    uint32_t nbOcts = data.extent(2);

    DataArrayScalar mach_number = DataArrayScalar("mach_number", nbCellsPerOct*nbOcts);

    Kokkos::parallel_for(
      nbOcts, KOKKOS_LAMBDA(uint32_t iOct) {
        for (uint32_t iCell = 0; iCell < nbCellsPerOct; ++iCell) {
          
          real_t d = datah(iCell, fm[ID], iOct);
          real_t u = datah(iCell, fm[IU], iOct) / datah(iCell, fm[ID], iOct);
          real_t v = datah(iCell, fm[IV], iOct) / datah(iCell, fm[ID], iOct);
          real_t w = fm[IW] == -1 ? 0 : 
            datah(iCell, fm[IW], iOct) / datah(iCell, fm[ID], iOct);
          
          // kinetic energy
          real_t eken = 0.5 * d * (u * u + v * v + w * w);
          
          // internal energy
          real_t eint = datah(iCell, fm[IE], iOct) - eken;
          
          // specific heat ratio
          real_t gamma0 = m_params.settings.gamma0;
          
          // pressure
          real_t p = (gamma0 - 1) * eint;
          
          // square speed of sound
          real_t cs2 = p / d;
          
          // velocity square
          real_t u2 = u * u + v * v + w * w;
          
          mach_number(iCell + nbCellsPerOct*iOct) = sqrt(u2 / cs2);
          
        } // end for iCell
      });
    
    // actual data writing
    write_attribute("Mach", mach_number.ptr_on_device(), 
                    0, IO_CELL_SCALAR,
                    H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE);

  }
    
  return 0;

} // HDF5_Writer::write_quadrant_mach_number

// =======================================================
// =======================================================
void
HDF5_Writer::io_hdf5_writev(hid_t fd, 
                            const std::string &name, 
                            void *data,
                            hid_t dtype_id, 
                            hid_t wtype_id, 
                            hid_t rank,
                            hsize_t dims[], 
                            hsize_t count[],
                            hsize_t start[])
{
  int                 status;
  UNUSED(status);
  hsize_t             size = 1;
  hid_t               filespace = 0;
  hid_t               memspace = 0;
  hid_t               dataset = 0;
  hid_t               dataset_properties = 0;
  hid_t               write_properties = 0;

  //CANOP_GLOBAL_INFOF("Writing \"%s\" of size %llu / %llu\n",
  //                   name.c_str(), count[0], dims[0]);

  // compute size of the dataset
  for (int i = 0; i < rank; ++i) {
    size *= count[i];
  }

  // create the layout in the file and 
  // in the memory of the current process
  filespace = H5Screate_simple(rank, dims, nullptr);
  memspace = H5Screate_simple(rank, count, nullptr);

  // set some properties
  dataset_properties = H5Pcreate(H5P_DATASET_CREATE);
  write_properties = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(write_properties, H5FD_MPIO_COLLECTIVE);

  // create the dataset and the location of the local data
  dataset = H5Dcreate2(fd, name.c_str(), wtype_id, filespace,
		       H5P_DEFAULT, dataset_properties, H5P_DEFAULT);
  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

  if (dtype_id != wtype_id) {
    status = H5Tconvert(dtype_id, wtype_id, size, data, nullptr, H5P_DEFAULT);
    //SC_CHECK_ABORT(status >= 0, "H5Tconvert failed!");
  }
  H5Dwrite(dataset, wtype_id, memspace, filespace, write_properties, data);

  H5Dclose(dataset);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(dataset_properties);
  H5Pclose(write_properties);

} // HDF5_Writer::io_hdf5_writev

// =======================================================
// =======================================================
void
HDF5_Writer::io_hdf5_write_coordinates()
{

  if (m_write_block_data) {

    //m_local_num_quads  = m_amr_mesh->getNumOctants();
    //m_global_num_quads = m_amr_mesh->getGlobalNumOctants();
    
    //m_local_num_nodes  = m_nbNodesPerCell * m_local_num_quads;
    //m_global_num_nodes = m_nbNodesPerCell * m_global_num_quads;

    uint32_t nbNodesPerCell = m_params.dimType==TWO_D ?
      (m_bx+1) * (m_by+1)            :
      (m_bx+1) * (m_by+1) * (m_bz+1);
    uint64_t totalNumOfCoords = 3 * m_local_num_quads * (m_bx+1) * (m_by+1) * (m_bz+1);

    std::vector<float> data(totalNumOfCoords);

    /*
     * construct the list of node coordinates
     */

    // total size
    real_t Lx = m_params.xmax - m_params.xmin;
    real_t Ly = m_params.ymax - m_params.ymin;
    real_t Lz = m_params.zmax - m_params.zmin;

    for (uint32_t i = 0; i < m_local_num_quads; ++i) {

      // retrieve cell size and rescale
      real_t cellSize = m_amr_mesh->getSize(i);

      real_t dx = cellSize/(m_bx)*Lx;
      real_t dy = cellSize/(m_by)*Ly;
      real_t dz = m_bz==0 ? 0 : cellSize/(m_bz)*Lz;

      // coordinates of the lower left corner
      real_t orig_x = m_amr_mesh->getNode(i, 0)[0];
      real_t orig_y = m_amr_mesh->getNode(i, 0)[1];
      real_t orig_z = m_amr_mesh->getNode(i, 0)[2];

      int inode = 0;
      for (int32_t jz = 0; jz < m_bz+1; ++jz) {
        for (int32_t jy = 0; jy < m_by+1; ++jy) {
          for (int32_t jx = 0; jx < m_bx+1; ++jx) {
            
            data[3 * (nbNodesPerCell * i + inode) + 0] = orig_x + jx*dx;
            data[3 * (nbNodesPerCell * i + inode) + 1] = orig_y + jy*dy;
            data[3 * (nbNodesPerCell * i + inode) + 2] = orig_z + jz*dz;

            ++inode;

          } // end for jx
        } // end for jy
      } // end for jz
    } // end for i

    // get prepared for hdf5 writing
    
    hsize_t dims[2] = {0, 0};
    hsize_t count[2] = {0, 0};
    hsize_t start[2] = {0, 0};
    
    // get the dimensions and offset of the node coordinates array
    dims[0] = m_global_num_quads * nbNodesPerCell;
    dims[1] = 3;

    count[0] = m_local_num_quads * nbNodesPerCell;
    count[1] = 3;

    // get global index of the first octant of current mpi processor
    start[0] = m_amr_mesh->getGlobalIdx((uint32_t)0) * nbNodesPerCell;
    start[1] = 0;

    // write the node coordinates
    io_hdf5_writev(this->m_hdf5_file, "coordinates", &(data[0]),
                   H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, 2 /* rank = 2 */, dims,
                   count, start);

  } else {

    std::vector<float> data(3 * m_local_num_nodes);

    /*
     * construct the list of node coordinates
     */

    for (uint32_t i = 0; i < m_local_num_quads; ++i) {

      for (uint8_t j = 0; j < m_nbNodesPerCell; ++j) {
        data[3 * m_nbNodesPerCell * i + 3 * j + 0] =
            m_amr_mesh->getNode(i, j)[0];
        data[3 * m_nbNodesPerCell * i + 3 * j + 1] =
            m_amr_mesh->getNode(i, j)[1];
        data[3 * m_nbNodesPerCell * i + 3 * j + 2] =
            m_amr_mesh->getNode(i, j)[2];
      }
    }

    // get prepared for hdf5 writing

    hsize_t dims[2] = {0, 0};
    hsize_t count[2] = {0, 0};
    hsize_t start[2] = {0, 0};

    // get the dimensions and offset of the node coordinates array
    dims[0] = m_global_num_nodes;
    dims[1] = 3;

    count[0] = m_local_num_nodes;
    count[1] = 3;

    // get global index of the first octant of current mpi processor
    start[0] = m_nbNodesPerCell * m_amr_mesh->getGlobalIdx((uint32_t)0);
    start[1] = 0;

    // write the node coordinates
    io_hdf5_writev(this->m_hdf5_file, "coordinates", &(data[0]),
                   H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, 2 /* rank = 2 */, dims,
                   count, start);
  }

} // HDF5_Writer::io_hdf5_write_coordinates

// =======================================================
// =======================================================
void HDF5_Writer::io_hdf5_write_connectivity()
{

  if (m_write_block_data) {

    int nbNodesPerLeaf = m_params.dimType == TWO_D ?
      (m_bx+1) * (m_by+1) :
      (m_bx+1) * (m_by+1) * (m_bz+1) ;

    std::vector<int> data(m_local_num_quads * m_nbCellsPerLeaf * m_nbNodesPerCell);
    
    // first node of current MPI process
    uint32_t globalNodeOffset = 
      nbNodesPerLeaf * 
      m_amr_mesh->getGlobalIdx((uint32_t)0);

    uint32_t nbConnectivityPerLeaf = m_nbCellsPerLeaf * m_nbNodesPerCell;

    // get connectivity data
    for (uint32_t iLeaf = 0; iLeaf < m_local_num_quads; ++iLeaf) {
      
      uint32_t localNodeOffset = 
        nbNodesPerLeaf *
        iLeaf;
      
      // sweep subcells
      int nz = m_params.dimType==2 ? 1 : m_bz;
      for (int jz = 0; jz < nz; ++jz) {
        for (int jy = 0; jy < m_by; ++jy) {
          for (int jx = 0; jx < m_bx; ++jx) {

            int nodeOffset = globalNodeOffset + localNodeOffset;

            uint32_t idx =  nbConnectivityPerLeaf * iLeaf +
              subCellIndex(jx,jy,jz) * m_nbNodesPerCell;
            
            data[idx + 0] = nodeOffset + subNodeIndex(jx  ,jy  ,jz);
            data[idx + 1] = nodeOffset + subNodeIndex(jx+1,jy  ,jz);
            data[idx + 2] = nodeOffset + subNodeIndex(jx+1,jy+1,jz);
            data[idx + 3] = nodeOffset + subNodeIndex(jx  ,jy+1,jz);

            if (m_params.dimType == THREE_D) {
              
              data[idx + 4] = nodeOffset + subNodeIndex(jx  ,jy  ,jz+1);
              data[idx + 5] = nodeOffset + subNodeIndex(jx+1,jy  ,jz+1);
              data[idx + 6] = nodeOffset + subNodeIndex(jx+1,jy+1,jz+1);
              data[idx + 7] = nodeOffset + subNodeIndex(jx  ,jy+1,jz+1);
              
            }
              
          } // end for jx
        } // end for jy
      } // end for jz

    } // end for iLeaf

    // now write connectivity with hdf5

    hsize_t dims[2]  = { 0, 0 };
    hsize_t count[2] = { 0, 0 };
    hsize_t start[2] = { 0, 0 };

    // get the dimensions and offsets for each connectivity array
    dims[0] = m_global_num_quads*m_nbCellsPerLeaf;
    dims[1] = m_nbNodesPerCell;

    count[0] = m_local_num_quads*m_nbCellsPerLeaf;
    count[1] = m_nbNodesPerCell;

    start[0] = m_amr_mesh->getGlobalIdx((uint32_t)0)*m_nbCellsPerLeaf;
    start[1] = 0;

    // write the node coordinates
    io_hdf5_writev(m_hdf5_file, "connectivity", &(data)[0], 
                   H5T_NATIVE_INT,
                   H5T_NATIVE_INT, 
                   2 /* rank */, 
                   dims, count, start);

  } else { // regular AMR mesh, i.e. one cell per leaf

    uint32_t node[8] =  {0, 1, 3, 2, 0, 0, 0, 0};
    
    if (m_amr_mesh->getDim() == 3) {
      node[4] = 4;
      node[5] = 5;
      node[6] = 7;
      node[7] = 6;
    }

    std::vector<int> data(m_local_num_quads*m_nbNodesPerCell);
    
    uint32_t in = 0;

    // get connectivity data
    for (uint32_t i = 0; i < m_local_num_quads; ++i) {

      for (int j = 0; j < m_nbNodesPerCell; ++j) {
      
        uint32_t idx = m_nbNodesPerCell * i + j;
      
        data[idx] = m_nbNodesPerCell * m_amr_mesh->getGlobalIdx((uint32_t) 0) + in + node[j];
      } // end for j
      
      in += m_nbNodesPerCell;
      
    } // end for i

    // now write connectivity with hdf5

    hsize_t dims[2]  = { 0, 0 };
    hsize_t count[2] = { 0, 0 };
    hsize_t start[2] = { 0, 0 };

    // get the dimensions and offsets for each connectivity array
    dims[0] = m_global_num_quads;
    dims[1] = m_nbNodesPerCell;

    count[0] = m_local_num_quads;
    count[1] = m_nbNodesPerCell;

    start[0] = m_amr_mesh->getGlobalIdx((uint32_t)0);
    start[1] = 0;

    // write the node coordinates
    io_hdf5_writev(m_hdf5_file, "connectivity", &(data)[0], 
                   H5T_NATIVE_INT,
                   H5T_NATIVE_INT, 
                   2 /* rank */, 
                   dims, count, start);

  }

} // HDF5_Writer::io_hdf5_write_connectivity

// =======================================================
// =======================================================
void
HDF5_Writer::io_hdf5_write_level()
{

  if (!this->m_write_level) {
    return;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<int> data(nbData);

  // gather level for each local quadrant
  uint32_t i=0;
  for (uint32_t iLeaf = 0; iLeaf < m_local_num_quads; ++iLeaf) {
    for (uint32_t j = 0; j < m_nbCellsPerLeaf; ++j) {
      data[i] = m_amr_mesh->getLevel(iLeaf);
      ++i;
    }
  }

  this->write_attribute("level", &(data)[0], 0, IO_CELL_SCALAR,
        		H5T_NATIVE_INT, H5T_NATIVE_INT);

} // HDF5_Writer::io_hdf5_write_level

// =======================================================
// =======================================================
void
HDF5_Writer::io_hdf5_write_iOct()
{

  if (!this->m_write_iOct) {
    return;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<uint32_t> data(nbData);

  // gather level for each local quadrant
  uint32_t i=0;
  for (uint32_t iLeaf = 0; iLeaf < m_local_num_quads; ++iLeaf) {
    for (uint32_t j = 0; j < m_nbCellsPerLeaf; ++j) {
      data[i] = iLeaf;
      ++i;
    }
  }

  this->write_attribute("iOct", &(data)[0], 0, IO_CELL_SCALAR,
        		H5T_NATIVE_UINT, H5T_NATIVE_UINT);

} // HDF5_Writer::io_hdf5_write_iOct

// =======================================================
// =======================================================
void
HDF5_Writer::io_hdf5_write_rank()
{

  if (!this->m_write_rank) {
    return;
  }

  uint32_t nbData = m_local_num_quads * m_nbCellsPerLeaf;

  std::vector<int> data(nbData);

  // gather rank for each local quadrant
  for (uint32_t i = 0; i < nbData; ++i) {
    data[i] = m_mpiRank;
  }
  
  this->write_attribute("rank", &(data)[0], 0, IO_CELL_SCALAR,
         		H5T_NATIVE_INT, H5T_NATIVE_INT);


} // HDF5_Writer::io_hdf5_write_rank

// =======================================================
// =======================================================
// Private members
// =======================================================
// =======================================================

// =======================================================
// =======================================================
void
HDF5_Writer::io_xdmf_write_main_header()
{

  FILE *fd = m_main_xdmf_file;

  fprintf(fd, "<?xml version=\"1.0\" ?>\n");
  fprintf(fd, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
  fprintf(fd, "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\""
	  " Version=\"2.0\">\n");
  fprintf(fd, "  <Domain Name=\"MainTimeSeries\">\n");
  fprintf(fd, "    <Grid Name=\"MainTimeSeries\" GridType=\"Collection\""
	  " CollectionType=\"Temporal\">\n");
  
} // HDF5_Writer::io_xdmf_write_main_header

// =======================================================
// =======================================================
void
HDF5_Writer::io_xdmf_write_header(double time)
{

  FILE   *fd = this->m_xdmf_file;
  size_t  global_num_cells = this->m_amr_mesh->getGlobalNumOctants();
  size_t  global_num_nodes = global_num_cells * m_nbNodesPerCell;

  if (m_write_block_data) {

    global_num_cells *= m_nbCellsPerLeaf;

    int nbNodesPerLeaf = m_params.dimType == TWO_D ?
      (m_bx+1) * (m_by+1) :
      (m_bx+1) * (m_by+1) * (m_bz+1) ;

    global_num_nodes = this->m_amr_mesh->getGlobalNumOctants() * nbNodesPerLeaf;
  }  

  const std::string IO_TOPOLOGY_TYPE = m_params.dimType == TWO_D ?
    IO_TOPOLOGY_TYPE_2D :
    IO_TOPOLOGY_TYPE_3D;

  fprintf(fd, "<?xml version=\"1.0\" ?>\n");
  fprintf(fd, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
  fprintf(fd, "<Xdmf Version=\"2.0\">\n");
  fprintf(fd, "  <Domain>\n");
  fprintf(fd, "    <Grid Name=\"%s\" GridType=\"Uniform\">\n", this->m_basename.c_str());
  fprintf(fd, "      <Time TimeType=\"Single\" Value=\"%g\" />\n", time);

  // Connectivity
  fprintf(fd, "      <Topology TopologyType=\"%s\" NumberOfElements=\"%lu\""
	  ">\n", IO_TOPOLOGY_TYPE.c_str(), global_num_cells);
  fprintf(fd, "        <DataItem Dimensions=\"%lu %d\" DataType=\"Int\""
	  " Format=\"HDF\">\n", global_num_cells, m_nbNodesPerCell);
  fprintf(fd, "         %s.h5:/connectivity\n", this->m_basename.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Topology>\n");

  // Points
  fprintf(fd, "      <Geometry GeometryType=\"XYZ\">\n");
  fprintf(fd, "        <DataItem Dimensions=\"%lu 3\" %s Format=\"HDF\">\n",
	  global_num_nodes, IO_XDMF_NUMBER_TYPE);
  fprintf(fd, "         %s.h5:/coordinates\n", this->m_basename.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Geometry>\n");

} // HDF5_Writer::io_xdmf_write_header

// =======================================================
// =======================================================
void
HDF5_Writer::io_xdmf_write_main_include(const std::string &name)
{

  FILE *fd = this->m_main_xdmf_file;

  fprintf(fd, "      <xi:include href=\"%s\""
	  " xpointer=\"xpointer(//Xdmf/Domain/Grid)\" />\n", name.c_str());

} // HDF5_Writer::io_xdmf_write_main_include

// =======================================================
// =======================================================
void
HDF5_Writer::io_xdmf_write_attribute(const std::string &name,
                                     const std::string &number_type,
                                     io_attribute_type_t type,
                                     hsize_t dims[2])
{

  FILE              *fd = this->m_xdmf_file;
  const std::string &basename = this->m_basename;
  
  fprintf(fd, "      <Attribute Name=\"%s\"", name.c_str());
  switch(type) {
  case IO_CELL_SCALAR:
    fprintf(fd, " AttributeType=\"Scalar\" Center=\"Cell\">\n");
    break;
  case IO_CELL_VECTOR:
    fprintf(fd, " AttributeType=\"Vector\" Center=\"Cell\">\n");
    break;
  case IO_NODE_SCALAR:
    fprintf(fd, " AttributeType=\"Scalar\" Center=\"Node\">\n");
    break;
  case IO_NODE_VECTOR:
    fprintf(fd, " AttributeType=\"Vector\" Center=\"Node\">\n");
    break;
  default:
    std::cerr << "Unsupported field type.\n";
    return;
  }

  fprintf(fd, "        <DataItem %s Format=\"HDF\"", number_type.c_str());
  if (type == IO_CELL_SCALAR || type == IO_NODE_SCALAR) {
    fprintf(fd, " Dimensions=\"%llu\">\n", dims[0]);
  }
  else {
    fprintf(fd, " Dimensions=\"%llu %llu\">\n", dims[0], dims[1]);
  }

  fprintf(fd, "         %s.h5:/%s\n", basename.c_str(), name.c_str());
  fprintf(fd, "        </DataItem>\n");
  fprintf(fd, "      </Attribute>\n");

} // HDF5_Writer::io_xdmf_write_attribute

// =======================================================
// =======================================================
void
HDF5_Writer::io_xdmf_write_main_footer()
{
  
  FILE *fd = this->m_main_xdmf_file;

  fprintf(fd, "    </Grid>\n");
  fprintf(fd, "  </Domain>\n");
  fprintf(fd, "</Xdmf>\n");
  
} // HDF5_Writer::io_xdmf_write_main_footer

// =======================================================
// =======================================================
void
HDF5_Writer::io_xdmf_write_footer()
{
  
  FILE *fd = this->m_xdmf_file;
  
  fprintf(fd, "    </Grid>\n");
  fprintf(fd, "  </Domain>\n");
  fprintf(fd, "</Xdmf>\n");
  
} // HDF5_Writer::io_xdmf_write_footer

// =======================================================
// =======================================================
int32_t
HDF5_Writer::subCellIndex(int jx, int jy, int jz) {

  return jx + m_bx * ( jy + m_by * jz);

} // HDF5_Writer::subCellIndex

// =======================================================
// =======================================================
int32_t
HDF5_Writer::subNodeIndex(int jx, int jy, int jz) {

  return jx + (m_bx+1) * ( jy + (m_by+1) * jz);

} // HDF5_Writer::subNodeIndex

} // namespace dyablo
