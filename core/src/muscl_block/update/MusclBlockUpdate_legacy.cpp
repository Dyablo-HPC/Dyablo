#include "muscl_block/update/MusclBlockUpdate_legacy.h"

#include "muscl_block/legacy/utils_block.h"
#include "muscl_block/legacy/CopyInnerBlockCellData.h"
#include "muscl_block/legacy/CopyGhostBlockCellData.h"
#include "muscl_block/legacy/ConvertToPrimitivesHydroFunctor.h"

#include "utils/monitoring/Timers.h"

#include "muscl_block/legacy/MusclBlockGodunovUpdateFunctor.h"

namespace dyablo { 
namespace muscl_block {

struct MusclBlockUpdate_legacy::Data{
  ForeachCell& foreach_cell;
  uint32_t nbOctsPerGroup;  
  int ndim;
  MusclBlockGodunovUpdateFunctor::Params params;
  BoundaryConditionType boundary_xmin, boundary_xmax;
  BoundaryConditionType boundary_ymin, boundary_ymax;
  BoundaryConditionType boundary_zmin, boundary_zmax;
  
  Timers& timers;  
};

MusclBlockUpdate_legacy::MusclBlockUpdate_legacy(
  ConfigMap& configMap,
  ForeachCell& foreach_cell,
  Timers& timers )
 : pdata(new Data
    {foreach_cell, 
    foreach_cell.pmesh.getNumOctants(),
    foreach_cell.pmesh.getDim(),
    MusclBlockGodunovUpdateFunctor::Params(configMap),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING),
    timers})
{
  pdata->nbOctsPerGroup = std::min( 
      foreach_cell.pmesh.getNumOctants(), 
      configMap.getValue<uint32_t>("amr","nbOctsPerGroup",foreach_cell.pmesh.getNumOctants()));

}

MusclBlockUpdate_legacy::~MusclBlockUpdate_legacy()
{}

void MusclBlockUpdate_legacy::update(const ForeachCell::CellArray_global_ghosted& Uin,
                                     const ForeachCell::CellArray_global_ghosted& Uout_, 
                                     real_t dt)
{
  static_assert( std::is_same<decltype(Uin.U), DataArrayBlock>::value, 
                 "MusclBlockUpdate_legacy can only be compiled for block-based ForeachCell" );
  
  ForeachCell& foreach_cell = pdata->foreach_cell;
  const LightOctree& lmesh = foreach_cell.pmesh.getLightOctree();
  const id2index_t& fm = Uin.fm;

  uint32_t nbOctsPerGroup = foreach_cell.cdata.nbOctsPerGroup;
  uint32_t bx = Uin.bx, by = Uin.by, bz = Uin.bz;  
  Timers& timers = pdata->timers; 
  GravityType gravity_type = pdata->params.gravity_type;
  real_t gamma0 = pdata->params.riemann_params.gamma0;
  real_t smallr = pdata->params.riemann_params.smallr;
  real_t smallp = pdata->params.riemann_params.smallp;

  uint32_t ghostWidth = 2;

  uint32_t nbOcts = lmesh.getNumOctants();
  uint32_t nbGroup = (nbOcts + nbOctsPerGroup - 1) / nbOctsPerGroup;
  
  uint32_t nbFields = fm.nbfields();
  uint32_t bx_g = bx + 2*ghostWidth;
  uint32_t by_g = by + 2*ghostWidth;
  uint32_t bz_g = bz + 2*ghostWidth;

  const DataArrayBlock& U = Uin.U;
  const DataArrayBlock& Ughost = Uin.Ughost;
  const DataArrayBlock& Uout = Uout_.U;
  DataArrayBlock Ugroup("Ugroup", bx_g*by_g*bz_g, nbFields, nbOctsPerGroup);
  DataArrayBlock Qgroup("Qgroup", bx_g*by_g*bz_g, nbFields, nbOctsPerGroup);

  InterfaceFlags interface_flags(nbOctsPerGroup);

  for (uint32_t iGroup = 0; iGroup < nbGroup; ++iGroup) 
  {
    timers.get("block copy").start();

    // Copy data from U to Ugroup
    CopyInnerBlockCellDataFunctor::apply( {pdata->ndim, gravity_type},
                                       fm,
                                       {bx,by,bz},
                                       ghostWidth,
                                       nbOcts,
                                       nbOctsPerGroup,
                                       U, Ugroup, 
                                       iGroup);
    CopyGhostBlockCellDataFunctor::apply(lmesh,
                                        {
                                          pdata->boundary_xmin, pdata->boundary_xmax,
                                          pdata->boundary_ymin, pdata->boundary_ymax,
                                          pdata->boundary_zmin, pdata->boundary_zmax,
                                          gravity_type
                                        },
                                        fm,
                                        {bx,by,bz},
                                        ghostWidth,
                                        nbOctsPerGroup,
                                        U,
                                        Ughost,
                                        Ugroup, 
                                        iGroup,
                                        interface_flags);

    timers.get("block copy").stop();

    // start main computation
    timers.get("godunov").start();

    // convert conservative variable into primitives ones for the given group
    ConvertToPrimitivesHydroFunctor::apply({
                                            pdata->ndim,
                                            gamma0, smallr, smallp,
                                          }, 
                                         fm,
                                         {bx,by,bz},
                                         ghostWidth,
                                         nbOcts,
                                         nbOctsPerGroup,
                                         iGroup,
                                         Ugroup, 
                                         Qgroup);

    // perform time integration :
    {
      MusclBlockGodunovUpdateFunctor::apply(lmesh,
                                          pdata->params,
                                          fm,
                                          {bx,by,bz},
                                          ghostWidth,
                                          nbOcts,
                                          nbOctsPerGroup,
                                          iGroup,
                                          Ugroup,
                                          U,
                                          Ughost,
                                          Uout,
                                          Qgroup,
                                          interface_flags,
                                          dt);
    }

    timers.get("godunov").stop();
  }
}

}// namespace dyablo
}// namespace muscl_block

FACTORY_REGISTER( dyablo::muscl_block::MusclBlockUpdateFactory, dyablo::muscl_block::MusclBlockUpdate_legacy, "MusclBlockUpdate_legacy" );

