#include "GravitySolver_constant.h"

#include "utils/monitoring/Timers.h"

#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"


namespace dyablo { 
namespace muscl_block {

struct GravitySolver_constant::Data{
  ForeachCell& foreach_cell;
  
  Timers& timers;  

  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;
  real_t gx, gy, gz;
};

GravitySolver_constant::GravitySolver_constant(
  ConfigMap& configMap,
  ForeachCell& foreach_cell,
  Timers& timers )
 : pdata(new Data
    {foreach_cell,
    timers,
    configMap.getValue<real_t>("mesh", "xmin", 0.0),
    configMap.getValue<real_t>("mesh", "ymin", 0.0),
    configMap.getValue<real_t>("mesh", "zmin", 0.0),
    configMap.getValue<real_t>("mesh", "xmax", 1.0),
    configMap.getValue<real_t>("mesh", "ymax", 1.0),
    configMap.getValue<real_t>("mesh", "zmax", 1.0),
    configMap.getValue<real_t>("gravity", "gx",  0.0),
    configMap.getValue<real_t>("gravity", "gy",  0.0),
    configMap.getValue<real_t>("gravity", "gz",  0.0)
    })
{}

GravitySolver_constant::~GravitySolver_constant()
{}

void GravitySolver_constant::update_gravity_field(
  const ForeachCell::CellArray_global_ghosted& Uin,
  const ForeachCell::CellArray_global_ghosted& Uout )
{
  using CellIndex = ForeachCell::CellIndex;

  uint8_t ndim = pdata->foreach_cell.getDim();
  real_t gx = pdata->gx, gy = pdata->gy, gz = pdata->gz;

  ForeachCell& foreach_cell = pdata->foreach_cell;

  pdata->timers.get("GravitySolver_constant").start();

  foreach_cell.foreach_patch( "GravitySolver_constant::update_gravity_field",
    PATCH_LAMBDA( const ForeachCell::Patch& patch )
  {
    patch.foreach_cell(Uout, CELL_LAMBDA(const CellIndex& iCell_Uout)
    {
      Uout.at(iCell_Uout, IGX) = gx;
      Uout.at(iCell_Uout, IGY) = gy;
      if(ndim == 3) Uout.at(iCell_Uout, IGZ) = gz;
    });
  });

  pdata->timers.get("GravitySolver_constant").stop();
}

}// namespace dyablo
}// namespace muscl_block

FACTORY_REGISTER( dyablo::muscl_block::GravitySolverFactory, dyablo::muscl_block::GravitySolver_constant, "GravitySolver_constant" );
