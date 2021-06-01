#include "muscl_block/update/MusclBlockUpdate_legacy.h"

#include "muscl_block/utils_block.h"
#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyGhostBlockCellData.h"
#include "muscl_block/ConvertToPrimitivesHydroFunctor.h"

#include "utils/monitoring/Timers.h"

#include "muscl_block/update/MusclBlockGodunovUpdateFunctor.h"

namespace dyablo { 
namespace muscl_block {

struct MusclBlockUpdate_legacy::Data{
  const ConfigMap& configMap;
  const HydroParams& params;  
  AMRmesh& pmesh;
  const id2index_t fm;
  uint32_t nbOctsPerGroup;  
  uint32_t bx, by, bz;    
  
  Timers& timers;  
};

MusclBlockUpdate_legacy::MusclBlockUpdate_legacy(
  const ConfigMap& configMap,
  const HydroParams& params, 
  AMRmesh& pmesh,
  const id2index_t& fm,
  uint32_t bx, uint32_t by, uint32_t bz,
  Timers& timers )
 : pdata(new Data
    {configMap, 
    params, 
    pmesh, 
    fm,
    pmesh.getNumOctants(),
    bx, by, bz,
    timers})
{
  pdata->nbOctsPerGroup = std::min( 
      pmesh.getNumOctants(), 
      (uint32_t)configMap.getInteger("amr","nbOctsPerGroup",pmesh.getNumOctants()));

}

MusclBlockUpdate_legacy::~MusclBlockUpdate_legacy()
{}

void MusclBlockUpdate_legacy::update(DataArrayBlock U, DataArrayBlock Ughost,
                                     DataArrayBlock Uout, 
                                     real_t dt)
{
  
  const ConfigMap& configMap = pdata->configMap;
  const HydroParams& params = pdata->params;  
  const LightOctree& lmesh = pdata->pmesh.getLightOctree();
  const id2index_t& fm = pdata->fm;

  uint32_t nbOctsPerGroup = pdata->nbOctsPerGroup;  
  uint32_t bx = pdata->bx, by = pdata->by, bz = pdata->bz;  
  Timers& timers = pdata->timers; 

  uint32_t ghostWidth = 2;

  uint32_t nbOcts = lmesh.getNumOctants();
  uint32_t nbGroup = (nbOcts + nbOctsPerGroup - 1) / nbOctsPerGroup;
  
  uint32_t nbFields = U.extent(1);
  uint32_t bx_g = bx + 2*ghostWidth;
  uint32_t by_g = by + 2*ghostWidth;
  uint32_t bz_g = bz + 2*ghostWidth;
  DataArrayBlock Ugroup("Ugroup", bx_g*by_g*bz_g, nbFields, nbOctsPerGroup);
  DataArrayBlock Qgroup("Qgroup", bx_g*by_g*bz_g, nbFields, nbOctsPerGroup);

  InterfaceFlags interface_flags(nbOctsPerGroup);

  for (uint32_t iGroup = 0; iGroup < nbGroup; ++iGroup) 
  {
    timers.get("block copy").start();

    // Copy data from U to Ugroup
    CopyInnerBlockCellDataFunctor::apply(configMap, params, fm,
                                       {bx,by,bz},
                                       ghostWidth,
                                       nbOcts,
                                       nbOctsPerGroup,
                                       U, Ugroup, 
                                       iGroup);
    CopyGhostBlockCellDataFunctor::apply(lmesh,
                                        configMap,
                                        params,
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
    ConvertToPrimitivesHydroFunctor::apply(configMap,
                                         params, 
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
                                          configMap,
                                          params,
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

