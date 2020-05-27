#include "utils/io/FileHandler.h"

namespace dyablo
{
namespace io
{

// =======================================================
// =======================================================
FileHandler::FileHandler() :
  directory("./"), name("data"), suffix("txt") {} // FileHandler::FileHandler

// =======================================================
// =======================================================
FileHandler::FileHandler(std::string directory, std::string name,
                         std::string suffix) :
  directory(directory), name(name), suffix(suffix)
{
} // FileHandler::FileHandler

// =======================================================
// =======================================================
FileHandler::~FileHandler() {} // FileHandler::~FileHandler

// =======================================================
// =======================================================
std::string FileHandler::getFullPath()
{

  std::stringstream filename;

  if (!directory.empty())
    filename << directory << "/";
  filename << name;

  filename << "." << suffix;

  return filename.str();

} // FileHandler::getFullPath

} // namespace io

} // namespace dyablo
