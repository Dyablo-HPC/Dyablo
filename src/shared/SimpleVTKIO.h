#ifndef SIMPLE_VTK_IO_H_
#define SIMPLE_VTK_IO_H_

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"
#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"

#include "shared/bitpit_common.h"

namespace euler_pablo {

/**
 * Use configMap information to retrieve a list of scalar field to write, 
 * compute the their id the access them in DataArray. This routine returns
 * a map with this information.
 *
 * \param[in,out] map       this is the map to fill
 * \param[in]     params    a DataParams reference object
 * \param[in]     configMap to access parameters settings
 *
 * \return the map size (i.e. the number of valid variable names)
 */
int build_var_to_write_map(str2int_t        & map,
			   const HydroParams& params,
			   const ConfigMap  & configMap);

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
void writeVTK(AMRmesh         &amr_mesh,
	      std::string      filenameSuffix,
	      DataArray        data,
	      id2index_t       fm,
	      str2int_t        names2index,
	      const ConfigMap& configMap);

/**
 * Write a  Kokkos::View<double*> (see also ParaTree::writeTest)
 */
void writeTest(AMRmesh               &amr_mesh,
	       std::string            filenameSuffix,
	       Kokkos::View<double*>  data);
  
} // namespace euler_pablo

#endif // SIMPLE_VTK_IO_H_
