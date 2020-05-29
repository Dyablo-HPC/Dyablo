#include "utils/io/FileHandlerVtk.h"

namespace dyablo
{
namespace io
{

// =======================================================
// =======================================================
FileHandlerVtk::FileHandlerVtk() :
  FileHandler(), timeStep(0), isParallel(false), mpiRank(0)
{

  suffix = "vtu";

} // FileHandlerVtk::FileHandlerVtk

// =======================================================
// =======================================================
FileHandlerVtk::FileHandlerVtk(std::string directory, std::string name,
                               std::string suffix) :
  FileHandler(directory, name, suffix),
  timeStep(0),
  isParallel(false),
  mpiRank(0)
{

} // FileHandlerVtk::FileHandlerVtk

// =======================================================
// =======================================================
FileHandlerVtk::~FileHandlerVtk() {} // FileHandlerVtk::~FileHandlerVtk

// =======================================================
// =======================================================
std::string FileHandlerVtk::getFullPath()
{

  std::stringstream filename;

  if (!directory.empty())
    filename << directory << "/";
  filename << name;

  // write timeStep in string timeFormat
  std::ostringstream timeFormat;
  timeFormat.width(7);
  timeFormat.fill('0');
  timeFormat << timeStep;
  filename << "_time";
  filename << timeFormat.str();

  if (isParallel)
  {
    // write MPI rank in string rankFormat
    std::ostringstream rankFormat;
    rankFormat.width(5);
    rankFormat.fill('0');
    rankFormat << mpiRank;
    filename << "_mpi";
    filename << rankFormat.str();
  }

  filename << "." << suffix;

  return filename.str();

} // FileHandlerVtk::getFullPath

} // namespace io

} // namespace dyablo
