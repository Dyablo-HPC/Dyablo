#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include <fstream>
#include <vector>

#include "enums.h"
#include "real_type.h"
#include "utils/config/ConfigMap.h"
#include "utils/io/FileHandlerVtk.h"

#include "kokkos_shared.h"

namespace dyablo
{
namespace io
{

enum class VTK_WRITE_ENUM
{

  ASCII    = 0,
  APPENDED = 1,
  BINARY   = 2

};

// ==================================================================
// ==================================================================
/**
 * \class VTKWriter using VTK unstructured grid format.
 *
 * If run in parallel (MPI enabled), each MPI process writes a vtu file,
 * and process 0 (master) writes an additionnal pvtu (partitionned vtu)
 * which enables Paraview or Visit to re-compose all the pieces together.
 *
 */
class VTKWriter
{

protected:
  FileHandlerVtk m_fileHandler; /**< file handler (directory, name, ...) */
  ConfigMap &configMap;         /**< application config map */

  int64_t m_nbCells;

  uint64_t m_offsetBytes = 0; /**< offset in bytes, only meaningful when
                                 binary ouput is enabled */

  int m_mpi_rank; /**< MPI rank of current process */
  int m_nb_ranks; /**< total number of MPI processes */

  VTK_WRITE_ENUM
      m_vtk_write_type;         /**< ascii, appended binary or base64 binary */
  std::string m_write_type_str; /**< a string identifying the type of vtk file
                                   format (ascii, appended or binary) */

  bool m_parallelEnabled; /**< write one piece per MPI process */

  std::fstream m_out_file; /**< iostream handle */

public:
  VTKWriter(ConfigMap &configMap, int64_t nbCells);
  virtual ~VTKWriter();

  void parallel_enable() { m_parallelEnabled = true; };
  void parallel_disable() { m_parallelEnabled = false; };

  void set_vtk_write_type(VTK_WRITE_ENUM wt) { m_vtk_write_type = wt; };
  bool vtk_ascii_enabled()
  {
    return m_vtk_write_type == VTK_WRITE_ENUM::ASCII;
  };
  bool vtk_appended_enabled()
  {
    return m_vtk_write_type == VTK_WRITE_ENUM::APPENDED;
  };
  bool vtk_binary_enabled()
  {
    return m_vtk_write_type == VTK_WRITE_ENUM::BINARY;
  };

  void set_mpi_rank(int rank) { m_mpi_rank = rank; };
  int get_mpi_rank() { return m_mpi_rank; };

  void set_nb_ranks(int nb_ranks) { m_nb_ranks = nb_ranks; };
  int get_nb_ranks() { return m_nb_ranks; };

  void open_file();
  void close_file();

  void write_header();
  void write_footer();

  /** Write scalar information cycle (=iStep) and time. */
  void write_metadata(int iStep, real_t time);

  /** Write number of nodes and cells for current piece (MPI sub-domain). */
  void write_piece_header(int64_t nbNodes);

  /** Just close the piece XML section. */
  void write_piece_footer();

  /** Write some scalar cell-centered data */
  void
  write_cell_data(const std::string &dataname,
                  const std::vector<real_t> &cell_data = std::vector<real_t>());

  /** Write some scalar cell-centered data from a Kokkos device array */
  void write_cell_data(const std::string &dataname,
                       const Kokkos::View<real_t *, Device> cell_data_d);

  /** open data section */
  void open_data();

  /** close data section */
  void close_data();

  /** close UnstructuredGrid section (that was open in write_header) */
  void close_grid();

  /** open section AppendedData - only useful for binary output */
  void open_data_appended();

  /** close section AppendedData - only useful for binary output */
  void close_data_appended();

  /** typedef Point holding coordinates of a point. */
  template<int dim>
  using Point = std::array<real_t, dim>;

  /**
   * Write geometry (nodes coordinates and cells connectivity).
   *
   * Cells are assumed quadrangle in 2D and hexaedral in 3D.
   *
   * When writing with binary appended format, nodes_locations is
   * not required yet.
   */
  template <int dim>
  void write_geometry(
      const std::vector<Point<3>> &nodes_location = std::vector<Point<3>>());

  /**
   * Write connectivity (list of nodes id, i.e. vertex associated to a cell.
   *
   * Cells are assumed quadrangle in 2D and hexahedral in 3D.
   *
   * VERY important: we assume an implicit connectivity; more precisely, all
   * vertex are redundant (we will do better later)
   *
   */
  template <int dim>
  void write_connectivity();

  /**
   * write nodes location, connectivity and offsets in binary format
   * at the end of the file, after the ascii header.
   */
  template <int dim>
  void write_appended_binary_geometry(std::vector<Point<3>> &nodes_location);

  /**
   * Write binary data in appended format; this toutine can be called
   * several times but in the same order as write_cell_data.
   */
  void write_appended_binary_cell_data(const std::string &dataname,
                                       const std::vector<real_t> &cell_data);

  /**
   * write binary data using base64 code. this code has been adapted from
   * libsc (LGPL, copyright C. Burstedde, https://github.com/cburstedde/libsc).
   */
  int write_base64_binary_data(const char *bytes, size_t data_size);

}; // class VTKWriter

// =======================================================
// =======================================================
template <int dim>
void VTKWriter::write_geometry(const std::vector<Point<3>> &nodes_location)
{

  uint64_t nbNodes = nodes_location.size();

  // 4 nodes for a quadrangle in 2d
  // 8 nodes for a hexahedron in 3d
  int nbNodesPerCell = (dim == 2) ? 4 : 8;

  m_out_file << "  <Points>\n";
  m_out_file << "    <DataArray type=\"Float32\" Name=\"Points\" "
                "NumberOfComponents=\"3\" format=\""
             << m_write_type_str << "\"";

  if (vtk_appended_enabled())
  {
    m_out_file << " offset=\"" << 0 << "\"";
  }

  m_out_file << ">"
             << "\n";

  /*
   * for binary format, see write_appended_binary_geometry
   */
  if (vtk_ascii_enabled())
  {

    for (uint64_t i = 0; i < nbNodes; ++i)
    {

      m_out_file << nodes_location[i][IX] << " " << nodes_location[i][IY] << " "
                 << nodes_location[i][IZ] << "\n";

    } // end for
  }
  else if (vtk_binary_enabled())
  {

    // this is only necessary for binary output
    float *vertices = new float[m_nbCells * nbNodesPerCell * 3];

    for (int64_t i = 0; i < m_nbCells * nbNodesPerCell; ++i)
    {

      vertices[3 * i]     = (float)nodes_location[i][IX];
      vertices[3 * i + 1] = (float)nodes_location[i][IY];
      vertices[3 * i + 2] = (float)nodes_location[i][IZ];

    } // end for i

    m_out_file << "          ";

    write_base64_binary_data(reinterpret_cast<const char *>(&(vertices[0])),
                             sizeof(float) * m_nbCells * nbNodesPerCell * 3);

    delete[] vertices;

    m_out_file << "\n";

  } // vtk_ascii_enabled or vtk_binary_enabled

  m_out_file << "    </DataArray>\n";
  m_out_file << "  </Points>\n";

  m_offsetBytes +=
      sizeof(uint64_t) + sizeof(float) * m_nbCells * nbNodesPerCell * 3;

} // VTKWriter::write_geometry

// =======================================================
// =======================================================
template <int dim>
void VTKWriter::write_connectivity()
{

  // 4 nodes for a quadrangle in 2d
  // 8 nodes for a hexahedron in 3d
  int nbNodesPerCell = (dim == 2) ? 4 : 8;

  // 9 means "Quad" - 12 means "Hexahedron"
  int cellType = (dim == 2) ? 9 : 12;

  m_out_file << "  <Cells>\n";

  /*
   * CONNECTIVITY
   */
  m_out_file << "    <DataArray type=\"Int64\" Name=\"connectivity\" format=\""
             << m_write_type_str << "\"";

  if (vtk_appended_enabled())
  {
    m_out_file << " offset=\"" << m_offsetBytes << "\"";
  }

  m_out_file << " >\n";

  m_offsetBytes +=
      sizeof(uint64_t) + sizeof(uint64_t) * m_nbCells * nbNodesPerCell;

  if (vtk_ascii_enabled())
  {

    for (int64_t i = 0; i < m_nbCells; ++i)
    {

      // offset to the first nodes in this cell
      uint64_t offset = i * nbNodesPerCell;
      for (uint8_t index = 0; index < nbNodesPerCell; ++index)
      {
        m_out_file << offset + index << " ";
      }

      m_out_file << "\n";
    }

  } // end vtk_ascii_enabled

  if (vtk_binary_enabled())
  {

    int64_t *offsets_array = new int64_t[m_nbCells * nbNodesPerCell];

    uint64_t tmp = 0;
    for (int64_t i = 0; i < m_nbCells; ++i)
    {

      // offset to the first nodes in this cell
      uint64_t offset = i * nbNodesPerCell;
      for (uint8_t index = 0; index < nbNodesPerCell; ++index)
      {
        offsets_array[tmp] = offset + index;
        ++tmp;
      }
    }

    m_out_file << "          ";

    write_base64_binary_data(
        reinterpret_cast<const char *>(&(offsets_array[0])),
        sizeof(int64_t) * m_nbCells * nbNodesPerCell);

    delete[] offsets_array;

    m_out_file << "\n";

  } // end vtk_binary_enabled

  m_out_file << "    </DataArray>\n";

  /*
   * OFFSETS
   */
  m_out_file << "    <DataArray type=\"Int64\" Name=\"offsets\" format=\""
             << m_write_type_str << "\"";

  if (vtk_appended_enabled())
  {
    m_out_file << " offset=\"" << m_offsetBytes << "\"";
  }

  m_out_file << " >\n";

  m_offsetBytes += sizeof(uint64_t) + sizeof(uint64_t) * m_nbCells;

  if (vtk_ascii_enabled())
  {
    // number of nodes per cell is 4 in 2D, 8 in 3D
    for (int64_t i = 1; i <= m_nbCells; ++i)
    {
      int64_t cell_offset = nbNodesPerCell * i;
      m_out_file << cell_offset << " ";
    }
    m_out_file << "\n";
  }

  if (vtk_binary_enabled())
  {

    int64_t *tmpArray = new int64_t[m_nbCells];

    for (int64_t i = 0; i < m_nbCells; ++i)
    {

      tmpArray[i] = nbNodesPerCell * (i + 1);
    }

    m_out_file << "          ";

    write_base64_binary_data(reinterpret_cast<const char *>(&(tmpArray[0])),
                             sizeof(int64_t) * m_nbCells);

    delete[] tmpArray;

    m_out_file << "\n";

  } // end vtk_binary_enabled

  m_out_file << "    </DataArray>\n";

  /*
   * CELL TYPES
   */
  m_out_file << "    <DataArray type=\"UInt8\" Name=\"types\" format=\""
             << m_write_type_str << "\"";

  if (vtk_appended_enabled())
  {
    m_out_file << " offset=\"" << m_offsetBytes << "\"";
  }

  m_out_file << " >\n";

  m_offsetBytes += sizeof(uint64_t) + sizeof(unsigned char) * m_nbCells;

  if (vtk_ascii_enabled())
  {
    for (int64_t i = 0; i < m_nbCells; ++i)
    {
      m_out_file << cellType << " ";
    }

    m_out_file << "\n";
  }

  if (vtk_binary_enabled())
  {

    uint8_t *tmpArray = new uint8_t[m_nbCells];

    for (int64_t i = 0; i < m_nbCells; ++i)
    {
      tmpArray[i] = (uint8_t)cellType;
    }

    m_out_file << "          ";

    write_base64_binary_data(reinterpret_cast<const char *>(&(tmpArray[0])),
                             sizeof(uint8_t) * m_nbCells);

    delete[] tmpArray;

    m_out_file << "\n";

  } // end vtk_binary_enabled

  m_out_file << "    </DataArray>\n";

  /*
   * Close Cells section.
   */
  m_out_file << "  </Cells>\n";

} // VTKWriter::write_connectivity

// ================================================================
// ================================================================
template <int dim>
void VTKWriter::write_appended_binary_geometry(
    std::vector<Point<3>> &nodes_location)
{

  // 4 nodes for a quadrangle in 2d
  // 8 nodes for a hexahedron in 3d
  int nbNodesPerCell = (dim == 2) ? 4 : 8;

  // total number of nodes (should be equal to nodes_location size)
  int64_t nbNodesTotal = m_nbCells * nbNodesPerCell;

  /*
   * Write nodes location.
   */
  {
    // this is only necessary for binary output
    std::vector<float> vertices;

    for (int64_t i = 0; i < nbNodesTotal; ++i)
    {

      vertices.push_back(nodes_location[i][IX]);
      vertices.push_back(nodes_location[i][IY]);
      vertices.push_back(nodes_location[i][IZ]);

    } // end for i

    uint64_t size = sizeof(float) * nbNodesTotal * 3;
    m_out_file.write(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    m_out_file.write(reinterpret_cast<char *>(&(vertices[0])), size);

    vertices.clear();

  } // end write nodes location

  /*
   * Write connectivity.
   */
  {
    // this is only necessary for binary output
    std::vector<uint64_t> connectivity;

    for (int64_t i = 0; i < m_nbCells; ++i)
    {

      // offset to the first nodes in this cell
      uint64_t offset = i * nbNodesPerCell;
      for (uint8_t index = 0; index < nbNodesPerCell; ++index)
        connectivity.push_back(offset + index);

    } // for i

    uint64_t size = sizeof(uint64_t) * m_nbCells * nbNodesPerCell;
    m_out_file.write(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    m_out_file.write(reinterpret_cast<char *>(&(connectivity[0])), size);

    connectivity.clear();

  } // end write connectivity

  /*
   * Write offsets.
   */
  {
    std::vector<uint64_t> offsets;

    // number of nodes per cell is 4 in 2D
    for (int64_t i = 1; i <= m_nbCells; ++i)
    {
      offsets.push_back(4 * i);
    }

    uint64_t size = sizeof(uint64_t) * m_nbCells;
    m_out_file.write(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    m_out_file.write(reinterpret_cast<char *>(&(offsets[0])), size);
    offsets.clear();

  } // end write offsets

  /*
   * Write cell types.
   */
  {
    std::vector<unsigned char> celltypes;

    // 9 means "Quad" - 12 means "Hexahedron"
    int cellType = (dim == 2) ? 9 : 12;

    for (int64_t i = 0; i < m_nbCells; ++i)
    {
      celltypes.push_back(cellType);
    }

    uint64_t size = sizeof(unsigned char) * m_nbCells;
    m_out_file.write(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    m_out_file.write(reinterpret_cast<char *>(&(celltypes[0])), size);
    celltypes.clear();
  }

} // write_appended_binary_geometry

} // namespace io

} // namespace dyablo

#endif // VTK_WRITER_H
