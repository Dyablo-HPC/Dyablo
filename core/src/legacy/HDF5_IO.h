#ifndef HDF5_IO_H_
#define HDF5_IO_H_

#include <hdf5.h>
#include <hdf5_hl.h>

#include "utils/config/ConfigMap.h"
#include "FieldManager.h"

#include "kokkos_shared.h" // for DataArray, DataArrayHost

#include "amr/AMRmesh.h"

namespace dyablo {

/**
 * \brief The four types of supported attribute types.
 */
enum io_attribute_type_t
  {
    IO_CELL_SCALAR,  //!< A cell-centered scalar.
    IO_CELL_VECTOR,  //!< A cell-centered vector.
    IO_NODE_SCALAR,  //!< A node-centered scalar.
    IO_NODE_VECTOR   //!< A node-centered vector.
  };

/**
 *\brief HDF5 writer.
 */
class HDF5_Writer {

public:
  /**
   * \brief Constructor.
   *
   * The writer is has no opened files by default except a main xmf file
   * that contains a collection of all the files that are to be written
   * next as a `Temporal` Collection.
   *
   * \param [in] amr_mesh 
   * \param [in] ConfigMap
   */
  HDF5_Writer(AMRmesh* amr_mesh, 
	      ConfigMap& configMap);
  ~HDF5_Writer();

  /**
   * To be called as often as the mesh changes.
   */
  void                update_mesh_info();

  /**
   * \brief Open the HDF5 and XMF files for writing.
   *
   * Also includes this file inside the main xmf file, if needed.
   */
  void                open (std::string basename, std::string outDir);

  /**
   * \brief Close the HDF5 and XMF files.
   *
   * Also closes the main xmf file, if it was opened.
   */
  void                close ();

  /**
   * \brief Write the header for the XMF and HDF5 files.
   *
   * The header includes the node information, connectivity information and
   * the treeid, level or mpirank for each quadrant, if required.
   *
   * In the case of the XMF file, this will defined the topology and geometry
   * of the mesh and point to the relevant fields in the HDF5 file.
   */
  int                 write_header (double time);

  /**
   * \brief Write the XMF footer.
   *
   * Closes all the XML tags.
   */
  int                 write_footer ();


  /**
   * \brief Write a node-centered or cell-centered attribute.
   *
   * \param [in] w The writer.
   * \param [in] name The name of the attribute.
   * \param [in] data The data to be written.
   * \param [in] dim In the case of a vector, this is the dimension of each
   * element in the vector field.
   * \param [in] ftype The type of the attribute. See supported types in
   *  the io_attribute_type_t enum.
   * \param [in] dtype The type of the data we are writing. This is given as a
   * native HDF5 type. See the types defined in the H5Tpublic.h header.
   * \param [in] wtype The type of the data written to the file. The
   * conversion between the data type and the written data is handled
   * by HDF5.
   */
  int                 write_attribute (const std::string &name,
				       void *data,
				       size_t dim,
				       io_attribute_type_t ftype,
				       hid_t dtype, 
                                       hid_t wtype);
  
  /**
   * \brief Write all cell-centered scalar attributes.
   *
   * \param
   * \param [in] data a Kokkos View with the user data
   * \param [in] fm the field map
   * \param [in] names2indes a std::map for names (string) to index(int) 
   * \param [in] type_id The HDF5 type id of the data to be written (double, int, ...). 
   *
   */
  int            write_quadrant_attribute (DataArrayHost datah,
                                           id2index_t    fm,
                                           str2int_t     names2index);

  /**
   * special variant of write_quadrant_attribute for velocity
   * vector field.
   */
  int            write_quadrant_velocity(DataArrayHost  datah,
                                         id2index_t     fm,
                                         bool           use_momentum);

  /**
   * special variant of write_quadrant_attribute for Mach number.
   *
   * Input array is assumed to contain conservative variables.
   */
  int            write_quadrant_mach_number(DataArrayHost datah,
                                            id2index_t    fm);

  /**
   * \brief Write all cell-centered scalar attributes when block amr is enabled.
   *
   * \param
   * \param [in] data a Kokkos View with the user data
   * \param [in] fm the field map
   * \param [in] names2indes a std::map for names (string) to index(int) 
   * \param [in] type_id The HDF5 type id of the data to be written (double, int, ...). 
   *
   */
  int            write_quadrant_attribute (DataArrayBlockHost data,
                                           id2index_t         fm,
                                           str2int_t          names2index);

  /**
   * \brief Special variant of write_quadrant_attribute for Pressure
   */
  int            write_quadrant_pressure(DataArrayBlockHost data,
  					 id2index_t         fm);

  /**
   * \brief Special variant of write_quadrant_attribute for Pressure
   */
  int            write_quadrant_id(DataArrayBlockHost data);

  /**
   * special variant of write_quadrant_attribute for Mach number.
   *
   * Input array is assumed to contain conservative variables.
   */
  int            write_quadrant_mach_number(DataArrayBlockHost data,
                                            id2index_t         fm);

  AMRmesh* m_amr_mesh; //!<
  std::string    m_basename; //!< the base name of the two files
  //id2index_t     m_fm; //!< field manager object
  //str2int_t      m_names2index; //!< map from names to user data variables

  bool           m_write_mesh_info; //!< write mesh info (oct level, mpi proc, ...)

  bool           m_write_block_data; //!< if true, expect a DataArrayBlock object
  int            m_bx; //!< block size along x
  int            m_by; //!< block size along y
  int            m_bz; //!< block size along z

  uint8_t        m_nbNodesPerCell;

  uint32_t       m_nbCellsPerLeaf;

  hid_t          m_hdf5_file;     //!< HDF5 file descriptor
  FILE          *m_xdmf_file;     //!< XDMF file descriptor
  FILE          *m_main_xdmf_file; //!< main XDMF file descriptor

  double         m_scale;            // default 1.0
  bool           m_write_level;      // default write_mesh_info (false)
  bool           m_write_rank;       // default write_mesh_info (false) 

  bool           m_write_iOct;       // default false

  int            m_mpiRank;

  // store information about nodes for writing node data
  uint64_t       m_global_num_nodes;
  uint32_t       m_local_num_nodes;
  //uint64_t       m_start_nodes;

  uint64_t       m_global_num_quads;
  uint32_t       m_local_num_quads;

  int m_ndim;
  real_t m_gamma0;
  real_t m_xmin, m_ymin, m_zmin;
  real_t m_xmax, m_ymax, m_zmax;

private:

  /*
   * XMDF utilities.
   */

  /**
   * \brief Write the header of the main XMF file.
   */
  void io_xdmf_write_main_header();

  /**
   * \brief Write the XMF header information: topology and geometry.
   */
  void io_xdmf_write_header (double time);

  /**
   * \brief Write the include for the current file.
   */
  void io_xdmf_write_main_include (const std::string &name);

  /**
   * \brief Write information about an attribute.
   *
   * \param [in] fd   file descriptor for xmf file
   * \param [in] basename The basename.
   * \param [in] name The name of the attribute.
   * \param [in] tyep The type.
   * \param [in] dims The dimensions of the attribute. If it is a scalar, dims[1]
   * will be ignored.
   */
  void io_xdmf_write_attribute(const std::string &name,
                               const std::string &number_type,
                               io_attribute_type_t type,
                               hsize_t dims[2]);

  /**
   * \brief Close the remaining tags for the main file.
   */
  void io_xdmf_write_main_footer();

  /**
   * \brief Close all remaining tags.
   *
   * \param[in] fd xmff file descriptor
   */
  void io_xdmf_write_footer();

  /*
   * HDF5 utilities.
   */
  /**
   * \brief Write a given dataset into the HDF5 file.
   *
   * \param [in] fd An open file descriptor to a HDF5 file.
   * \param [in] name The name of the dataset we are writing.
   * \param [in] data The data to write.
   * \param [in] dtype_id The native HDF5 type of the given data.
   * \param [in] wtype_id The native HDF5 type of the written data.
   * \param [in] rank The rank of the dataset. 1 if it is a vector, 2 for a matrix.
   * \param [in] dims The global dimensions of the dataset.
   * \param [in] count The local dimensions of the dataset.
   * \param [in] start The offset of the local data with respect to the global
   * positioning.
   *
   * \sa H5TPublic.h
   */
  void
  io_hdf5_writev(hid_t fd, const std::string &name, void *data,
                 hid_t dtype_id, hid_t wtype_id, hid_t rank,
                 hsize_t dims[], hsize_t count[], hsize_t start[]);

  /**
   * \brief Compute and write the coordinates of all the mesh nodes.
   */
  void io_hdf5_write_coordinates();

  /**
   * \brief Compute and write the connectivity information for each quadrant.
   */
  void io_hdf5_write_connectivity();

  /**
   * \brief Compute and write the level for each quadrant.
   */
  void io_hdf5_write_level();

  /**
   * \brief Write local quadrant id (local to current MPI process).
   * Purely for debug purpose.
   */
  void io_hdf5_write_iOct();

  /**
   * \brief Compute and write the MPI rank for each quadrant.
   *
   * The rank is wrapped with IO_MPIRANK_WRAP.
   */
  void io_hdf5_write_rank();

  //!
  int32_t subCellIndex(int jx, int jy, int jz);

  //!
  int32_t subNodeIndex(int jx, int jy, int jz);

}; // class HDF5_Writer

} // namespace dyablo

#endif // HDF5_IO_H_
