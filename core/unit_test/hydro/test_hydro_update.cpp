#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>

#include "hydro/HydroUpdate.h"
#include "init/InitialConditions.h"
#include "utils/config/ConfigMap.h"
#include "legacy/HydroParams.h"

namespace bdata = boost::unit_test::data;

namespace dyablo {


void run_test(const std::string& update_name, std::string ini_string)
{
    char* ini_cstr = &ini_string[0]

    ConfigMap configmap(ini_cstr, ini_string.size());
    HydroParams params; params.setup(configmap);
    FieldManager fieldMgr(configmap);
    uint32_t bx = configMap.getInteger("amr","bx",4);
    uint32_t by = configMap.getInteger("amr","by",4);
    uint32_t bz = configMap.getInteger("amr","bz",4);

    AMRmesh amr_mesh;

    

    Timers timers;
    HydroUpdate& update = HydroUpdateFactory::make_instance( update_name,
                                    configMap,
                                    params,
                                    *amr_mesh, 
                                    fieldMgr.get_id2index(),
                                    bx, by, bz,
                                    timers
                                );
}  


} // namespace dyablo

BOOST_AUTO_TEST_SUITE(dyablo)
BOOST_AUTO_TEST_SUITE(muscl_block)

BOOST_DATA_TEST_CASE(test_update_hydro, 
     bdata::make({  "HydroUpdate_legacy",
                    "HydroUpdate_generic"}) 
   * bdata::make({  "blast", "four_quadrant" }),
    update_version, init_version)
{
    
}

BOOST_AUTO_TEST_SUITE_END() /* muscl_block */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */