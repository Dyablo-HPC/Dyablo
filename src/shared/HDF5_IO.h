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
	      std::shared_ptr<ConfigMap> configMap,
              std::shared_ptr<HydroParams> params);
  ~HDF5_Writer();

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

  std::shared_ptr<AMRmesh> m_amr_mesh; //!<
  std::string    m_basename; //!< the base name of the two files
  //id2index_t     m_fm; //!< field manager object
  //str2int_t      m_names2index; //!< map from names to user data variables

  std::shared_ptr<ConfigMap> m_configMap;
  std::shared_ptr<HydroParams> m_params;

  bool           m_write_mesh_info; //!< write mesh info (oct level, mpi proc, ...)

  uint8_t        m_nbNodesPerCell;

  hid_t          m_hdf5_file;     //!< HDF5 file descriptor
  FILE          *m_xdmf_file;     //!< XDMF file descriptor
  FILE          *m_main_xdmf_file; //!< main XDMF file descriptor

  double         m_scale;            // default 1.0
  bool           m_write_level;      // default 1
  bool           m_write_rank;       // default 1

  int            m_mpiRank;

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


}; // class HDF5_Writer

} // namespace dyablo

#endif // HDF5_IO_H_
