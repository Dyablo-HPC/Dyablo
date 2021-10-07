/**
 * \file ConfigMap.cpp
 * \brief Implement ConfigMap, essentially a INIReader with additional get methods.
 *
 * \date 12 November 2010
 * \author Pierre Kestener.
 *
 * $Id: ConfigMap.cpp 1783 2012-02-21 10:20:07Z pkestene $
 */
#include "ConfigMap.h"
#include <cstdlib> // for strtof
#include <sstream>
#include <fstream>

#include "utils/mpi/GlobalMpiSession.h"

// =======================================================
// =======================================================
ConfigMap::ConfigMap(std::string filename) :
  INIReader(filename)
{
} // ConfigMap::ConfigMap

// =======================================================
// =======================================================
ConfigMap::ConfigMap(char* &buffer, int buffer_size) :
  INIReader(buffer, buffer_size)
{
} // ConfigMap::ConfigMap

// =======================================================
// =======================================================
ConfigMap::~ConfigMap()
{
} // ConfigMap::~ConfigMap

// =======================================================
// =======================================================
float ConfigMap::getFloat(std::string section, std::string name, float default_value) const
{
  std::string valstr = getString(section, name, "");
  const char* value = valstr.c_str();
  char* end;
  // This parses "1234" (decimal) and also "0x4D2" (hex)
  float valFloat = strtof(value, &end);
  return end > value ? valFloat : default_value;
} // ConfigMap::getFloat

// =======================================================
// =======================================================
void ConfigMap::setFloat(std::string section, std::string name, float value)
{

  std::stringstream ss;
  ss << value;

  setString(section, name, ss.str());

} // ConfigMap::setFloat

// =======================================================
// =======================================================
bool ConfigMap::getBool(std::string section, std::string name, bool default_value) const
{
  bool val = default_value;
  std::string valstr = getString(section, name, "");
  
  if (!valstr.compare("1") or 
      !valstr.compare("yes") or 
      !valstr.compare("true") or
      !valstr.compare("on"))
    val = true;
  if (!valstr.compare("0") or 
      !valstr.compare("no") or 
      !valstr.compare("false") or
      !valstr.compare("off"))
    val=false;
  
  // if valstr is empty, return the default value
  if (!valstr.size())
    val = default_value;

  return val;

} // ConfigMap::getBool

// =======================================================
// =======================================================
void ConfigMap::setBool(std::string section, std::string name, bool value)
{

  if (value)
    setString(section, name, "true");
  else
    setString(section, name, "false");

} // ConfigMap::setBool

// =======================================================
// =======================================================
ConfigMap broadcast_parameters(std::string filename)
{
  const dyablo::MpiComm& mpi_comm = dyablo::GlobalMpiSession::get_comm_world();
  int myRank = mpi_comm.MPI_Comm_rank();
  
  char* buffer = nullptr;
  int buffer_size = 0;
  
  // MPI rank 0 reads parameter file
  if (myRank == 0) {
    
    // open file and go to the end to get file size in bytes
    std::ifstream filein(filename.c_str(), std::ifstream::ate);
    if( !filein )
      throw std::runtime_error("Could not open .ini file : `" + filename + "`");
    int file_size = filein.tellg();
    
    filein.seekg(0); // rewind
    
    buffer_size = file_size;
    buffer = new char[buffer_size];
    
    filein.read(buffer, buffer_size);
  }
  
  // broacast buffer size (collective)
  mpi_comm.MPI_Bcast(&buffer_size, 1, 0);

  // all other MPI task need to allocate buffer
  if (myRank>0) {
    //printf("I'm rank %d allocating buffer of size %d\n",myRank,buffer_size);
    buffer = new char[buffer_size];
  }

  // broastcast buffer itself (collective)
  mpi_comm.MPI_Bcast(&buffer[0], buffer_size, 0);
  
  // now all MPI rank should have buffer filled, try to build a ConfigMap
  ConfigMap configMap(buffer,buffer_size);

  if (buffer)
    delete [] buffer;

  return configMap;
    
} // broadcast_parameters

