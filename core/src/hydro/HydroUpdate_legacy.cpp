#include "hydro/HydroUpdate_legacy.h"

#include "legacy/utils_block.h"
#include "legacy/CopyInnerBlockCellData.h"
#include "legacy/CopyGhostBlockCellData.h"
#include "legacy/ConvertToPrimitivesHydroFunctor.h"
#include "legacy/LegacyDataArray.h"

#include "utils/monitoring/Timers.h"

#include "legacy/MusclBlockGodunovUpdateFunctor.h"

namespace dyablo { 


struct HydroUpdate_legacy::Data{
  ForeachCell& foreach_cell;
  uint32_t nbOctsPerGroup;  
  int ndim;
  MusclBlockGodunovUpdateFunctor::Params params;
  BoundaryConditionType boundary_xmin, boundary_xmax;
  BoundaryConditionType boundary_ymin, boundary_ymax;
  BoundaryConditionType boundary_zmin, boundary_zmax;
  
  Timers& timers;  
};

HydroUpdate_legacy::HydroUpdate_legacy(
  ConfigMap& configMap,
  ForeachCell& foreach_cell,
  Timers& timers )
 : pdata(new Data
    {foreach_cell, 
    configMap.getValue<uint32_t>("amr","nbOctsPerGroup",64),
    foreach_cell.get_amr_mesh().getDim(),
    MusclBlockGodunovUpdateFunctor::Params(configMap),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING),
    configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING),
    timers})
{
}

HydroUpdate_legacy::~HydroUpdate_legacy()
{}

void HydroUpdate_legacy::update( UserData& U_, ScalarSimulationData& scalar_data) 
{
  real_t dt = scalar_data.get<real_t>("dt");
  ForeachCell& foreach_cell = pdata->foreach_cell;
  const LightOctree& lmesh = foreach_cell.get_amr_mesh().getLightOctree();

  uint32_t nbOctsPerGroup = std::min( 
      lmesh.getNumOctants(), 
      pdata->nbOctsPerGroup
    );
  uint32_t bx = foreach_cell.blockSize()[IX]; 
  uint32_t by = foreach_cell.blockSize()[IY]; 
  uint32_t bz = foreach_cell.blockSize()[IZ];
  Timers& timers = pdata->timers; 
  GravityType gravity_type = pdata->params.gravity_type;
  bool has_gravity_field = (gravity_type & GRAVITY_FIELD);
  real_t gamma0 = pdata->params.riemann_params.gamma0;
  real_t smallr = pdata->params.riemann_params.smallr;
  real_t smallp = pdata->params.riemann_params.smallp;

  uint32_t ghostWidth = 2;

  uint32_t nbOcts = lmesh.getNumOctants();
  uint32_t nbGroup = (nbOcts + nbOctsPerGroup - 1) / nbOctsPerGroup;
  
  LegacyDataArray Uin(  has_gravity_field
                        ? 
                        U_.getAccessor({
                            {"rho", ID},
                            {"e_tot", IE},
                            {"rho_vx", IU},
                            {"rho_vy", IV},
                            {"rho_vz", IW},
                            {"gx", IGX},
                            {"gy", IGY},
                            {"gz", IGZ}
                        })
                        :
                        U_.getAccessor({
                            {"rho", ID},
                            {"e_tot", IE},
                            {"rho_vx", IU},
                            {"rho_vy", IV},
                            {"rho_vz", IW},
                        })
  );
  LegacyDataArray Uout( U_.getAccessor({
      {"rho_next", ID},
      {"e_tot_next", IE},
      {"rho_vx_next", IU},
      {"rho_vy_next", IV},
      {"rho_vz_next", IW}
  }));

  uint32_t nbFields = Uin.get_id2index().nbfields();
  uint32_t bx_g = bx + 2*ghostWidth;
  uint32_t by_g = by + 2*ghostWidth;
  uint32_t bz_g = bz + 2*ghostWidth;

  DataArrayBlock Ugroup("Ugroup", bx_g*by_g*bz_g, nbFields, nbOctsPerGroup);
  DataArrayBlock Qgroup("Qgroup", bx_g*by_g*bz_g, nbFields, nbOctsPerGroup);

  InterfaceFlags interface_flags(nbOctsPerGroup);

  id2index_t fm = Uin.get_id2index();

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
                                       Uin, Ugroup, 
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
                                        Uin,
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
                                          Uin,
                                          Uout,
                                          Qgroup,
                                          interface_flags,
                                          dt);
    }

    timers.get("godunov").stop();
  }
}

}// namespace dyablo


FACTORY_REGISTER( dyablo::HydroUpdateFactory, dyablo::HydroUpdate_legacy, "HydroUpdate_legacy" );

