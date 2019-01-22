#ifndef SIMPLE_VTK_IO_H_
#define SIMPLE_VTK_IO_H_

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"
#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"

#include "shared/bitpit_common.h"

namespace euler_pablo {

/**
 * Simple VTK IO routine.
 *
 * Here we assume DataArray size is the same as the number of AMRmesh octants.
 *
 * This adapted from ParaTree member writeTest to fit the Kokkos::View interface, and
 * modified to handle multi valued data.
 *
 * \param[in] amr_mesh a PabloUniform const reference
 * \param[in] filename a string specifiy the output filename suffix (e.g. nb of iter)
 * \param[in] data a Kokkos::View to the data to save
 * \param[in] fm a field map to access data
 * \param[in] configMap a ConfigMap object to access input parameter file data (ini file format)
 */
void writeVTK(AMRmesh& amr_mesh,
	      std::string filenameSuffix,
	      DataArray data,
	      id2index_t  fm,
	      const ConfigMap& configMap);
  
} // namespace euler_pablo

#endif // SIMPLE_VTK_IO_H_
