#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace dyablo
{
namespace io
{

// ==================================================================
// ==================================================================
/**
 * \class FileHandler
 * \brief Create file name.
 *
 * Does not handle the file descriptor, only the client code can do that.
 * This class is just a simple structure, holding a few parameters
 * to ease building the full filename.
 *
 */
class FileHandler
{

protected:
  std::string directory; /**< name of directory where file resides. */
  std::string name;      /**< name of file. */
  std::string suffix;    /**< suffix. */

public:
  FileHandler();
  FileHandler(std::string directory, std::string name, std::string suffix);
  virtual ~FileHandler();

  void setDirectory(std::string the_directory) { directory = the_directory; };
  void setName(std::string the_name) { name = the_name; };
  void setSuffix(std::string the_suffix) { suffix = the_suffix; };

  /**
   * This is where the full path is build.
   * It will be overloaded in derived class, taking into account
   * specific file numbering for a time serie.
   */
  virtual std::string getFullPath();

  std::string getName() const { return name; };
  std::string getDirectory() const { return directory; };
  std::string getSuffix() const { return suffix; };

}; // class FileHandler

} // namespace io

} // namespace dyablo

#endif // FILE_HANDLER_H
