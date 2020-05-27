#ifndef FILE_HANDLER_VTK_H
#define FILE_HANDLER_VTK_H

#include "utils/io/FileHandler.h"

namespace dyablo
{
namespace io
{

// =======================================================
// =======================================================
/**
 * \class FileHandlerVtk
 */
class FileHandlerVtk : public FileHandler
{

protected:
  int timeStep;    /**< identify a time step number */
  bool isParallel; /**< if true, the file name will contain the MPI rank */
  int mpiRank;     /**< only used if isParallel is true */

public:
  FileHandlerVtk();
  FileHandlerVtk(std::string directory, std::string name, std::string suffix);
  virtual ~FileHandlerVtk();

  void setTimeStep(int the_timeStep) { timeStep = the_timeStep; };
  void setIsParallel(bool the_isParallel) { isParallel = the_isParallel; };
  void setRank(int the_rank) { mpiRank = the_rank; };

  std::string getFullPath();

  bool IsParallel() const { return isParallel; };
  std::string getTimeStep() const { return directory; };
  int getRank() const { return mpiRank; };

}; // class FileHandlerVtk

} // namespace io

} // namespace dyablo

#endif // FILE_HANDLER_VTK_H
