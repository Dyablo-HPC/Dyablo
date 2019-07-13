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
