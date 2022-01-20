#include "GravitySolver_constant.h"

#include "muscl_block/utils_block.h"
#include "utils/monitoring/Timers.h"

#include "muscl_block/foreach_cell/ForeachCell.h"
#include "shared/utils_hydro.h"


namespace dyablo { 
namespace muscl_block {

struct GravitySolver_constant::Data{
  AMRmesh& pmesh;
  const id2index_t fm;
 
  uint32_t bx, by, bz; 
  
  Timers& timers;  

  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;
  real_t gx, gy, gz;
};

GravitySolver_constant::GravitySolver_constant(
  ConfigMap& configMap,
  std::shared_ptr<AMRmesh> pmesh,
  const id2index_t& fm,
  uint32_t bx, uint32_t by, uint32_t bz,
  Timers& timers )
 : pdata(new Data
    {*pmesh, 
    fm,
    bx, by, bz,
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
    DataArrayBlock U_, DataArrayBlock Ughost_,
    DataArrayBlock Uout_)
{
  using CellArray = ForeachCell::CellArray_global;
  using CellIndex = ForeachCell::CellIndex;

  uint8_t ndim = pdata->pmesh.getDim();
  const id2index_t& fm = pdata->fm;
  uint32_t bx = pdata->bx, by = pdata->by, bz = pdata->bz;
  real_t xmin = pdata->xmin, ymin = pdata->ymin, zmin = pdata->zmin;
  real_t xmax = pdata->xmax, ymax = pdata->ymax, zmax = pdata->zmax;
  real_t gx = pdata->gx, gy = pdata->gy, gz = pdata->gz;
  const uint32_t nbOctsPerGroup = pdata->pmesh.getNumOctants();

  ForeachCell foreach_cell(
    pdata->pmesh, 
    bx, by, bz, 
    xmin, ymin, zmin,
    xmax, ymax, zmax,
    nbOctsPerGroup
  );

  CellArray Uout = foreach_cell.get_global_array(Uout_, 0, 0, 0, fm);

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
