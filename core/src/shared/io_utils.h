#ifndef IO_UTILS_H_
#define IO_UTILS_H_

#include "shared/HydroParams.h"
#include "utils/config/ConfigMap.h"
#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"

#include "shared/amr/AMRmesh.h"

namespace dyablo {

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

} // namespace dyablo

#endif // IO_UTILS_H_
