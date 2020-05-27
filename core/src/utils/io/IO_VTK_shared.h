#ifndef IO_VTK_SHARED_H_
#define IO_VTK_SHARED_H_

#include <fstream>
#include <string>

#include "shared/real_type.h"
#include "utils/config/ConfigMap.h"
#include "utils/misc/utils.h"

namespace dyablo
{
namespace io
{

/**
 * Write VTK unstructured grid header.
 */
void write_vtu_header(std::ostream &outFile, ConfigMap &configMap);

/**
 * Write VTK unstructured grid metadata (date and time).
 */
void write_vtk_metadata(std::ostream &outFile, int iStep, real_t time);
/**
 * Write closing VTK unstructured grid statement.
 */
void close_vtu_grid(std::ostream &outFile);

/**
 * Write closing VTK file statement.
 */
void write_vtu_footer(std::ostream &outFile);

#ifdef DYABLO_USE_MPI
/**
 * write pvtu header in a separate file.
 */
void write_pvtu_header(std::string headerFilename, std::string outputPrefix,
                       const int nProcs, ConfigMap &configMap,
                       const std::map<int, std::string> &varNames,
                       const int iStep);
#endif // DYABLO_USE_MPI

} // namespace io

} // namespace dyablo

#endif // IO_VTK_SHARED_H_
