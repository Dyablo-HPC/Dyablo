#ifndef HDF5_IO_H_
#define HDF5_IO_H_

#include <hdf5.h>
#include <hdf5_hl.h>

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"
#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"

#include "shared/bitpit_common.h"

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
   * \param [in] fm filed manager
   * \param [in] names2index map variables names to index
   * \param [in] ConfigMap
   * \param [in] HydroParams
   *
   */
  HDF5_Writer(std::shared_ptr<AMRmesh> amr_mesh, 
	      //id2index_t       fm,
	      //str2int_t        names2index,
	      ConfigMap& configMap,
              HydroParams& params);
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
  void                open (std::string basename);

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
  

  std::shared_ptr<AMRmesh> m_amr_mesh; //!<
  std::string    m_basename; //!< the base name of the two files
  //id2index_t     m_fm; //!< field manager object
  //str2int_t      m_names2index; //!< map from names to user data variables

  ConfigMap&     m_configMap;
  HydroParams&   m_params;

  bool           m_write_mesh_info; //!< write mesh info (oct level, mpi proc, ...)

  uint8_t        m_nbNodesPerCell;

  hid_t          m_hdf5_file;     //!< HDF5 file descriptor
  FILE          *m_xdmf_file;     //!< XDMF file descriptor
  FILE          *m_main_xdmf_file; //!< main XDMF file descriptor

  double         m_scale;            // default 1.0
  bool           m_write_level;      // default 1
  bool           m_write_rank;       // default 1

  int            m_mpiRank;

  // store information about nodes for writing node data
  uint64_t       m_global_num_nodes;
  uint32_t       m_local_num_nodes;
  //uint64_t       m_start_nodes;

  uint64_t       m_global_num_quads;
  uint32_t       m_local_num_quads;

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
   * \brief Compute and write the MPI rank for each quadrant.
   *
   * The rank is wrapped with IO_MPIRANK_WRAP.
   */
  void io_hdf5_write_rank();

}; // class HDF5_Writer

} // namespace dyablo

#endif // HDF5_IO_H_
