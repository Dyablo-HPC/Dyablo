#ifndef SIMPLE_VTK_IO_H_
#define SIMPLE_VTK_IO_H_

#include <vector>

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"
#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"

#include "shared/amr/AMRmesh.h"

#include "shared/io_utils.h"

namespace dyablo {

/**
 * Simple VTK IO routine (simple means Partitionned VTU, using ASCII).
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
 * \param[in] names2index a map of names (of scalar field to save) to id (to fm)
 * \param[in] configMap a ConfigMap object to access input parameter file data (ini file format)
 */
void writeVTK(AMRmesh_pablo         &amr_mesh,
	      std::string      filenameSuffix,
	      DataArray        data,
	      id2index_t       fm,
	      str2int_t        names2index,
	      ConfigMap& configMap,
              std::string      nameSuffix = "");

/**
 * Write a  Kokkos::View<double*> (see also ParaTree::writeTest).
 */
void writeTest(AMRmesh_pablo               &amr_mesh,
	       std::string            filenameSuffix,
	       Kokkos::View<double*>  data);

/**
 * Write a std::vector<double> (see also ParaTree::writeTest).
 *
 * TODO : this routine is a duplicate from the previous one, only intended
 * for testing ideas when kokkos refactoring still underway (i.e. not
 * finished).
 */
void writeTest(AMRmesh_pablo               &amr_mesh,
	       std::string            filenameSuffix,
	       std::vector<double>    data);

} // namespace dyablo

#endif // SIMPLE_VTK_IO_H_
