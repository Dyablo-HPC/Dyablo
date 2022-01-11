#include <sstream>

#include "IO_VTK_shared.h"

#include "shared/enums.h"     // for ComponentIndex3D (IX,IY,IZ)
#include "utils/misc/utils.h"

namespace dyablo
{
namespace io
{

// =======================================================
// =======================================================
static bool isBigEndian()
{
  const int i = 1;
  return ((*(char *)&i) == 0);
}

// =======================================================
// =======================================================
void write_vtu_header(std::ostream &outFile, ConfigMap &configMap)
{

  bool outputVtkAscii = configMap.getValue<bool>("output", "outputVtkAscii", false);
  bool outputVtkAppended =
      configMap.getValue<bool>("output", "outputVtkAppended", false);
  bool outputVtkBinary = configMap.getValue<bool>("output", "outputVtkBinary", false);
  bool outputDateAndTime =
      configMap.getValue<bool>("output", "outputDateAndTime", false);

  // if writing raw binary data (file does not respect XML standard)
  if (outputVtkAscii or outputVtkBinary)
    outFile << "<?xml version=\"1.0\"?>\n";

  // print data and time
  outFile << "<!-- \n";
  outFile << "# vtk DataFile Version 3.0" << '\n'
          << "#This file was generated by dyablo";
  if (outputDateAndTime)
  {
    outFile << " on " << get_current_date();
  }
  else
  {
    outFile << ".";
  }
  outFile << "\n-->\n";

  // write xml data header

  outFile << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\"";

  if (isBigEndian())
  {
    outFile << " byte_order=\"BigEndian\"";
  }
  else
  {
    outFile << " byte_order=\"LittleEndian\"";
  }

  if (outputVtkAppended)
    outFile << " header_type=\"UInt64\"";

  outFile << ">\n";

  outFile << "  <UnstructuredGrid>\n";

} // write_vtu_header

// =======================================================
// =======================================================
void write_vtk_metadata(std::ostream &outFile, int iStep, real_t time)
{

  outFile << "  <FieldData>\n";

  outFile << "    <DataArray type=\"Int32\" Name=\"CYCLE\" "
             "NumberOfTuples=\"1\" format=\"ascii\">"
          << iStep << "    </DataArray>\n";

  outFile << "    <DataArray type=\"Float32\" Name=\"TIME\" "
             "NumberOfTuples=\"1\" format=\"ascii\">"
          << time << "    </DataArray>\n";

  outFile << "  </FieldData>\n";

} // write_vtk_metadata

// =======================================================
// =======================================================
void close_vtu_grid(std::ostream &outFile)
{

  outFile << "  </UnstructuredGrid>\n";

} // close_vtu_grid

// =======================================================
// =======================================================
void write_vtu_footer(std::ostream &outFile)
{

  outFile << "</VTKFile>\n";

} // write_vtu_footer

/*
 * write pvtu header in a separate file.
 */
// =======================================================
// =======================================================
void write_pvtu_header(std::string headerFilename, std::string outputPrefix,
                       const int nProcs, ConfigMap &configMap,
                       const std::map<int, std::string> &varNames,
                       const int iStep)
{
  // file handler
  std::fstream outHeader;

  bool outputDateAndTime =
      configMap.getValue<bool>("output", "outputDateAndTime", false);

  const int nbvar = varNames.size();

  // dummy string here, when using the full VTK API, data can be compressed
  // here, no compression used
  std::string compressor("");

  // check scalar data type
  bool useDouble       = sizeof(real_t) == sizeof(double) ? true : false;
  const char *dataType = useDouble ? "Float64" : "Float32";

  // write iStep in string timeFormat
  std::ostringstream timeFormat;
  timeFormat.width(7);
  timeFormat.fill('0');
  timeFormat << iStep;

  // open pvtu header file
  outHeader.open(headerFilename.c_str(), std::ios_base::out);

  outHeader << "<?xml version=\"1.0\"?>" << std::endl;

  // print data and time
  outHeader << "<!-- \n";
  outHeader << "# vtk DataFile Version 3.0" << '\n'
            << "#This file was generated by dyablo";
  if (outputDateAndTime)
  {
    outHeader << " on " << get_current_date();
  }
  else
  {
    outHeader << ".";
  }
  outHeader << "\n-->\n";

  if (isBigEndian())
    outHeader << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" "
                 "byte_order=\"BigEndian\" header_type=\"UInt64\" "
              << compressor << ">" << std::endl;
  else
    outHeader << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" "
                 "byte_order=\"LittleEndian\" header_type=\"UInt64\" "
              << compressor << ">" << std::endl;

  outHeader << "  <PUnstructuredGrid GhostLevel=\"0\">\n";

  outHeader << "    <PPoints>\n";
  outHeader
      << "      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>\n";
  outHeader << "    </PPoints>\n";

  outHeader << "    <PCells>\n";
  outHeader << "      <PDataArray type=\"Int64\" Name=\"connectivity\" "
               "NumberOfComponents=\"1\"/>\n";
  outHeader << "      <PDataArray type=\"Int64\" Name=\"offsets\"      "
               "NumberOfComponents=\"1\"/>\n";
  outHeader << "      <PDataArray type=\"UInt8\" Name=\"types\"        "
               "NumberOfComponents=\"1\"/>\n";
  outHeader << "    </PCells>\n";

  outHeader << "    <PCellData Scalars=\"Scalars_\">" << std::endl;
  for (int iVar = 0; iVar < nbvar; iVar++)
  {
    outHeader << "      <PDataArray type=\"" << dataType << "\" Name=\""
              << varNames.at(iVar) << "\"/>" << std::endl;
  }
  outHeader << "    </PCellData>" << std::endl;

  // one piece per MPI process
  for (int iPiece = 0; iPiece < nProcs; ++iPiece)
  {
    std::ostringstream pieceFormat;
    pieceFormat.width(5);
    pieceFormat.fill('0');
    pieceFormat << iPiece;
    std::string pieceFilename = outputPrefix + "_time" + timeFormat.str() +
                                "_mpi" + pieceFormat.str() + ".vtu";
    outHeader << "    <Piece Source=\"" << pieceFilename << "\"/>" << std::endl;
  }
  outHeader << "</PUnstructuredGrid>" << std::endl;
  outHeader << "</VTKFile>" << std::endl;

  // close header file
  outHeader.close();

  // end writing pvtu header

} // write_pvtu_header

} // namespace io

} // namespace dyablo
