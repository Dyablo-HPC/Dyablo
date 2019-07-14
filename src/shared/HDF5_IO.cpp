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
                         //id2index_t     fm,
                         //str2int_t      names2index,
                         ConfigMap& configMap,
                         HydroParams& params) :
  m_amr_mesh(amr_mesh),
  //m_fm(fm),
  //m_names2index(names2index),
  m_configMap(configMap),
  m_params(params)
{

  m_write_mesh_info = m_configMap.getBool("output", "write_mesh_info", false);

  m_write_level = m_write_mesh_info;
  m_write_rank =  m_write_mesh_info;

  m_nbNodesPerCell = m_params.dimType==TWO_D ? 
    IO_NODES_PER_CELL_2D : 
    IO_NODES_PER_CELL_3D;
  
  update_mesh_info();

  printf("%d %d %d %d\n",m_local_num_quads,m_global_num_quads,m_local_num_nodes,m_global_num_nodes);

  m_basename = ""; // TODO setup from params
  m_hdf5_file = 0;
  m_xdmf_file = nullptr;

  if (m_mpiRank == 0 and m_basename.size() ) {
    std::string filename;
    filename = m_basename + "_main.xmf";

    // INFOF("Writing main XMDF file \"%s\".\n", filename.c_str());
    m_main_xdmf_file = fopen(filename.c_str(), "w");
    io_xdmf_write_main_header();
  } else {
    m_main_xdmf_file = nullptr;
  }

  m_mpiRank = m_amr_mesh->getRank();

} // HDF5_Writer::HDF5_Writer

// =======================================================
// =======================================================
HDF5_Writer::~HDF5_Writer()
{
  
  /*
   * Only rank 0 needs to close the main xdmf file.
   */
  if (m_mpiRank == 0 and m_main_xdmf_file) {
    io_xdmf_write_main_footer();

    fflush(m_main_xdmf_file);
    fclose(m_main_xdmf_file);
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
  //io_hdf5_write_connectivity();
  //io_hdf5_write_level();
  //io_hdf5_write_rank();

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
  int                 mpirank = m_amr_mesh->getRank();

  hsize_t             dims[2] = { 0, 0 };
  hsize_t             count[2] = { 0, 0 };
  hsize_t             start[2] = { 0, 0 };
  int                 rank = 0;

  if (ftype == IO_CELL_SCALAR || ftype == IO_CELL_VECTOR) {

    dims[0] = m_amr_mesh->getGlobalNumOctants();
    dims[1] = dim;

    count[0] = m_amr_mesh->getNumOctants();
    count[1] = dims[1];

    // get global index of the first octant of current mpi processor
    start[0] = m_amr_mesh->getGlobalIdx((uint32_t) 0);
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
  if (mpirank == 0) {
    const char *dtype_str = hdf5_native_type_to_string(dtype);
    io_xdmf_write_attribute(name, dtype_str, ftype, dims);
  }

  io_hdf5_writev(m_hdf5_file, name, data, dtype, wtype, rank, dims, count, start);
  return 0;
  
} // HDF5_Writer::write_attribute

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

  hsize_t             dims[2] = { 0, 0 };
  hsize_t             count[2] = { 0, 0 };
  hsize_t             start[2] = { 0, 0 };

  // get the dimensions and offset of the node coordinates array
  dims[0] = m_global_num_nodes;
  dims[1] = 3;
  
  count[0] = m_local_num_nodes;
  count[1] = 3;
  
  // get global index of the first octant of current mpi processor
  start[0] = m_amr_mesh->getGlobalIdx((uint32_t) 0);
  start[1] = 0;

  // write the node coordinates
  io_hdf5_writev(this->m_hdf5_file, "coordinates", &(data[0]),
                 H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT, 2, 
                 dims, count, start);

} // HDF5_Writer::io_hdf_write_coordinates

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
  size_t  global_num_quads = this->m_amr_mesh->getGlobalNumOctants();
  size_t  global_num_nodes = global_num_quads * m_nbNodesPerCell;

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
	  ">\n", IO_TOPOLOGY_TYPE.c_str(), global_num_quads);
  fprintf(fd, "        <DataItem Dimensions=\"%lu %d\" DataType=\"Int\""
	  " Format=\"HDF\">\n", global_num_quads, m_nbNodesPerCell);
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

} // namespace dyablo
