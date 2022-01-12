#include "GravitySolver_constant.h"

#include "muscl_block/utils_block.h"
#include "utils/monitoring/Timers.h"

#include "muscl_block/foreach_cell/AMRBlockForeachCell_group.h"
#include "shared/utils_hydro.h"


namespace dyablo { 
namespace muscl_block {

struct GravitySolver_constant::Data{
  const HydroParams params;  
  AMRmesh& pmesh;
  const id2index_t fm;
 
  uint32_t bx, by, bz; 
  
  Timers& timers;  

  uint32_t nbOctsPerGroup;
};

GravitySolver_constant::GravitySolver_constant(
  ConfigMap& configMap,
  const HydroParams& params, 
  std::shared_ptr<AMRmesh> pmesh,
  const id2index_t& fm,
  uint32_t bx, uint32_t by, uint32_t bz,
  Timers& timers )
 : pdata(new Data
    {params, 
    *pmesh, 
    fm,
    bx, by, bz,
    timers
    })
{}

GravitySolver_constant::~GravitySolver_constant()
{}

void GravitySolver_constant::update_gravity_field(
    DataArrayBlock U_, DataArrayBlock Ughost_,
    DataArrayBlock Uout_)
{
  using ForeachCell = AMRBlockForeachCell_group;
  using CellArray = typename ForeachCell::CellArray_global;
  using CellIndex = typename ForeachCell::CellIndex;

  const HydroParams& params = pdata->params;
  const LightOctree& lmesh = pdata->pmesh.getLightOctree();
  uint8_t ndim = lmesh.getNdim();
  const id2index_t& fm = pdata->fm;
  uint32_t bx = pdata->bx, by = pdata->by, bz = pdata->bz;
  real_t xmin = params.xmin, ymin = params.ymin, zmin = params.zmin;
  real_t xmax = params.xmax, ymax = params.ymax, zmax = params.zmax;
  real_t gx = params.gx, gy = params.gy, gz = params.gz;
  const uint32_t nbOctsPerGroup = lmesh.getNumOctants();

  ForeachCell foreach_cell(
    ndim,
    lmesh, 
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
