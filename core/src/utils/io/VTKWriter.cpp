#include "utils/io/libb64/libb64.h" // for type base64 encoding
#include <algorithm>                // for std::max
#include <ios>                      // for std::ios::good

#include "utils/io/VTKWriter.h"

#include "utils/io/IO_VTK_shared.h"
#include "utils/mpi/GlobalMpiSession.h"

namespace dyablo
{
namespace io
{

// =======================================================
// =======================================================
VTKWriter::VTKWriter(ConfigMap &configMap, int64_t nbCells) :
  m_fileHandler(),
  configMap(configMap),
  m_nbCells(nbCells),
  m_mpi_rank(0),
  m_nb_ranks(1),
  m_vtk_write_type(VTK_WRITE_ENUM::ASCII),
  m_write_type_str("undefined"),
  m_parallelEnabled(false),
  m_out_file()
{
  // output directory
  std::string outputDir = configMap.getValue<std::string>("output", "outputDir", "./");
  m_fileHandler.setDirectory(outputDir);

  // file name base
  std::string outputPrefix =
      configMap.getValue<std::string>("output", "outputPrefix", "output");
  m_fileHandler.setName(outputPrefix);

  // file suffix (vtk unstructured grid)
  m_fileHandler.setSuffix("vtu");

  // set write type
  bool outputVtkAscii = configMap.getValue<bool>("output", "outputVtkAscii", false);
  bool outputVtkAppended =
      configMap.getValue<bool>("output", "outputVtkAppended", false);
  bool outputVtkBinary = configMap.getValue<bool>("output", "outputVtkBinary", true);

  if (outputVtkAscii)
  {
    m_vtk_write_type = VTK_WRITE_ENUM::ASCII;
  }
  else if (outputVtkAppended)
  {
    m_vtk_write_type = VTK_WRITE_ENUM::APPENDED;
  }
  else if (outputVtkBinary)
  {
    m_vtk_write_type = VTK_WRITE_ENUM::BINARY;
  }

  // set write type string
  if (outputVtkAscii)
    m_write_type_str = "ascii";
  else if (outputVtkAppended)
    m_write_type_str = "appended";
  else if (outputVtkBinary)
    m_write_type_str = "binary";

  const dyablo::MpiComm& mpi_comm = dyablo::GlobalMpiSession::get_comm_world();
  int rank = mpi_comm.MPI_Comm_rank();
  int nbProcs = mpi_comm.MPI_Comm_size();

  m_fileHandler.setIsParallel(nbProcs > 1);
  m_fileHandler.setRank(rank);
} // VTKWriter::VTKWriter

// =======================================================
// =======================================================
VTKWriter::~VTKWriter()
{

  // make sure to close output file
  close_file();

} // VTKWriter::~VTKWriter

// =======================================================
// =======================================================
void VTKWriter::open_file()
{

  // open file only if not already open
  if (!m_out_file.is_open())
    m_out_file.open(m_fileHandler.getFullPath(), std::ios_base::out);

} // VTKWriter::open_file

// =======================================================
// =======================================================
void VTKWriter::close_file()
{

  // close file only if currently open
  if (m_out_file.is_open())
    m_out_file.close();

} // VTKWriter::close_file

// =======================================================
// =======================================================
void VTKWriter::write_header()
{

  if (m_out_file.is_open())
    write_vtu_header(m_out_file, configMap);
  else
    std::cerr << "file " << m_fileHandler.getFullPath() << " is not open...\n";

} // VTKWriter::write_header

// =======================================================
// =======================================================
void VTKWriter::write_footer()
{

  if (m_out_file.is_open())
    write_vtu_footer(m_out_file);
  else
    std::cerr << "file " << m_fileHandler.getFullPath() << " is not open...\n";

} // VTKWriter::write_footer

// =======================================================
// =======================================================
void VTKWriter::write_metadata(int iStep, real_t time)
{

  if (m_out_file.is_open())
    write_vtk_metadata(m_out_file, iStep, time);
  else
    std::cerr << "file " << m_fileHandler.getFullPath() << " is not open...\n";

} // VTKWriter::write_metadata

// =======================================================
// =======================================================
void VTKWriter::write_piece_header(int64_t nbNodes)
{

  if (m_out_file.is_open())
  {

    m_out_file << "  <Piece NumberOfPoints=\"" << nbNodes
               << "\" NumberOfCells=\"" << m_nbCells << "\" >\n";
  }

} // VTKWriter::write_piece_header

// =======================================================
// =======================================================
void VTKWriter::write_piece_footer()
{

  if (m_out_file.is_open())
  {
    m_out_file << "  </Piece>\n";
  }

} // VTKWriter::write_piece_footer

// =======================================================
// =======================================================
void VTKWriter::write_cell_data(const std::string &dataname,
                                const std::vector<real_t> &cell_data)
{

  const char *dataType = "Float64";

  /*
   * write cell data.
   */
  m_out_file << "    <DataArray type=\"" << dataType << "\" Name=\"" << dataname
             << "\" format=\"" << m_write_type_str << "\"";

  if (vtk_appended_enabled())
  {
    m_out_file << " offset=\"" << m_offsetBytes << "\"";
  }

  m_out_file << " >\n";

  m_offsetBytes += sizeof(uint64_t) + sizeof(real_t) * m_nbCells;

  if (vtk_ascii_enabled())
  {

    for (int64_t i = 0; i < m_nbCells; ++i)
    {
      m_out_file << cell_data[i] << " ";
    }
    m_out_file << "\n";

  } // end vtk_ascii_enabled

  if (vtk_binary_enabled())
  {

    write_base64_binary_data(reinterpret_cast<const char *>(&(cell_data[0])),
                             sizeof(real_t) * m_nbCells);

    m_out_file << "\n";

  } // end vtk_binary_enabled

  m_out_file << "    </DataArray>\n";

} // VTKWriter::write_cell_data

// =======================================================
// =======================================================
void VTKWriter::write_cell_data(
    const std::string &dataname,
    const Kokkos::View<real_t *, Device> cell_data_d)
{

  Kokkos::View<real_t *, Device>::HostMirror cell_data =
      Kokkos::create_mirror(cell_data_d);

  Kokkos::deep_copy(cell_data, cell_data_d);

  const char *dataType = "Float64";

  /*
   * write cell data.
   */
  m_out_file << "    <DataArray type=\"" << dataType << "\" Name=\"" << dataname
             << "\" format=\"" << m_write_type_str << "\"";

  if (vtk_appended_enabled())
  {
    m_out_file << " offset=\"" << m_offsetBytes << "\"";
  }

  m_out_file << " >\n";

  m_offsetBytes += sizeof(uint64_t) + sizeof(real_t) * m_nbCells;

  if (vtk_ascii_enabled())
  {

    for (int64_t i = 0; i < m_nbCells; ++i)
    {
      m_out_file << cell_data(i) << " ";
    }
    m_out_file << "\n";

  } // end vtk_ascii_enabled

  if (vtk_binary_enabled())
  {

    write_base64_binary_data(reinterpret_cast<const char *>(cell_data.data()),
                             sizeof(real_t) * m_nbCells);

    m_out_file << "\n";

  } // end vtk_binary_enabled

  m_out_file << "    </DataArray>\n";

} // VTKWriter::write_cell_data

// =======================================================
// =======================================================
void VTKWriter::open_data()
{
  m_out_file << "  <CellData>\n";
} // VTKWriter::open_data

// =======================================================
// =======================================================
void VTKWriter::close_data()
{
  m_out_file << "  </CellData>\n";
} // VTKWriter::close_data

// =======================================================
// =======================================================
void VTKWriter::close_grid()
{

  close_vtu_grid(m_out_file);

} // VTKWriter::close_grid

// =======================================================
// =======================================================
void VTKWriter::open_data_appended()
{
  m_out_file << " <AppendedData encoding=\"raw\">"
             << "\n";

  // leading underscore
  m_out_file << "_";

} // VTKWriter::open_data_appended

// =======================================================
// =======================================================
void VTKWriter::close_data_appended()
{

  m_out_file << " </AppendedData>"
             << "\n";

} // VTKwriter::close_data_appended

// ================================================================
// ================================================================
void VTKWriter::write_appended_binary_cell_data(
    const std::string &dataname, 
    const std::vector<real_t> &cell_data)
{

  /*
   * Write cells data.
   */
  {

    uint64_t size = sizeof(real_t) * m_nbCells;
    m_out_file.write(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    m_out_file.write(reinterpret_cast<const char *>(&(cell_data[0])), size);

  } // end write cells data

} // VTKWriter::write_appended_binary_cell_data

// ================================================================
// ================================================================
int VTKWriter::write_base64_binary_data(const char *numeric_data,
                                        size_t byte_length)
{

  size_t chunks, chunksize, remaining, writenow;
  size_t code_length, base_length;
  uint32_t int_header;
  char *base_data;
  base64_encodestate encode_state;

  /* VTK format used 32bit header info */
  // assert (byte_length <= (size_t) UINT32_MAX);

  /* This value may be changed although this is not tested with VTK */
  chunksize  = (size_t)1 << 15; /* 32768 */
  int_header = (uint32_t)byte_length;

  /* Allocate sufficient memory for base64 encoder */
  code_length = 2 * std::max(chunksize, sizeof(int_header));
  code_length = std::max(code_length, (size_t)4) + 1;
  base_data   = new char[code_length];

  base64_init_encodestate(&encode_state);
  base_length = base64_encode_block((char *)&int_header,
                                    sizeof(int_header),
                                    base_data,
                                    &encode_state);
  // assert (base_length < code_length);
  base_data[base_length] = '\0';
  m_out_file.write(base_data, base_length);

  chunks    = 0;
  remaining = byte_length;
  while (remaining > 0)
  {
    writenow    = std::min(remaining, chunksize);
    base_length = base64_encode_block(numeric_data + chunks * chunksize,
                                      writenow,
                                      base_data,
                                      &encode_state);
    // assert (base_length < code_length);
    base_data[base_length] = '\0';
    m_out_file.write(base_data, base_length);
    remaining -= writenow;
    ++chunks;
  }

  base_length = base64_encode_blockend(base_data, &encode_state);
  // assert (base_length < code_length);
  base_data[base_length] = '\0';
  m_out_file.write(base_data, base_length);

  delete[] base_data;
  if (!m_out_file.good())
  {
    // something wrong happended !
    return -1;
  }
  return 0;

} // VTKWriter::write_base64_binary_data

} // namespace io

} // namespace dyablo
