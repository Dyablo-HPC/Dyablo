#pragma once

#include "muscl_block/MarkOctantsHydroFunctor.h"
#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyGhostBlockCellData.h"
#include "muscl_block/ConvertToPrimitivesHydroFunctor.h"

namespace dyablo {
namespace muscl_block {

class RefineCondition
{
  using CellArray = AMRBlockForeachCell_group::CellArray_global_ghosted;
public:
  RefineCondition( ConfigMap& configMap,
                const HydroParams& params, 
                AMRmesh& pmesh,
                const id2index_t& fm,
                uint32_t bx, uint32_t by, uint32_t bz,
                Timers& timers )
    : configMap(configMap),
      params(params),
      pmesh(pmesh),
      timers(timers),
      error_min ( configMap.getValue<real_t>("amr", "error_min", 0.2) ),
      error_max ( configMap.getValue<real_t>("amr", "error_max", 0.8) ),
      nbOctsPerGroup( configMap.getValue<uint32_t>("amr", "nbOctsPerGroup", 32) )
  {}

  void mark_cells( const CellArray& U )
  {
    real_t error_min = this->error_min;
    real_t error_max = this->error_max;
    uint32_t nbOctsPerGroup = this->nbOctsPerGroup;

    uint32_t bx = U.bx;
    uint32_t by = U.by;
    uint32_t bz = U.bz;
    constexpr int ghostWidth = 2; // with 2 ghosts in each side
    uint32_t nbCellsPerOct_g = (bx+2*ghostWidth)*(by+2*ghostWidth)*(bz+2*ghostWidth); 
    uint32_t nbfields = U.U.extent(1);

    auto fm = U.fm;

    DataArrayBlock Udata = U.U;
    DataArrayBlock Ugroup("Ugroup", nbCellsPerOct_g, nbfields, nbOctsPerGroup);
    DataArrayBlock Qgroup("Qgroup", nbCellsPerOct_g, nbfields, nbOctsPerGroup);
    InterfaceFlags interface_flags(nbOctsPerGroup);

    // apply refinement criterion group by group
    uint32_t nbOcts = pmesh.getNumOctants();;
    // number of group of octants, rounding to upper value
    uint32_t nbGroup = (nbOcts + nbOctsPerGroup - 1) / nbOctsPerGroup;

    MarkOctantsHydroFunctor::markers_t markers(nbOcts);

    for (uint32_t iGroup = 0; iGroup < nbGroup; ++iGroup) {

      timers.get("AMR: block copy").start();

      // Copy data from U to Ugroup
      CopyInnerBlockCellDataFunctor::apply(params, fm,
                                        {bx,by,bz},
                                        ghostWidth,
                                        nbOcts,
                                        nbOctsPerGroup,
                                        U.U, Ugroup, 
                                        iGroup);
      CopyGhostBlockCellDataFunctor::apply(pmesh.getLightOctree(),
                                          params,
                                          fm,
                                          {bx,by,bz},
                                          ghostWidth,
                                          nbOctsPerGroup,
                                          U.U,
                                          U.Ughost,
                                          Ugroup, 
                                          iGroup,
                                          interface_flags);

      timers.get("AMR: block copy").stop();

      timers.get("AMR: mark cells").start();

      // convert conservative variable into primitives ones for the given group
      ConvertToPrimitivesHydroFunctor::apply(params, 
                                          fm,
                                          {bx,by,bz},
                                          ghostWidth,
                                          nbOcts,
                                          nbOctsPerGroup,
                                          iGroup,
                                          Ugroup, 
                                          Qgroup);

      // finaly apply refine criterion : 
      // call device functor to flag for refine/coarsen
      MarkOctantsHydroFunctor::apply(pmesh.getLightOctree(), params, fm,
                                    {bx,by,bz}, ghostWidth,
                                    nbOcts, nbOctsPerGroup,
                                    Qgroup, iGroup,
                                    error_min, error_max,
                                    markers);

      timers.get("AMR: mark cells").stop();

    } // end for iGroup

    MarkOctantsHydroFunctor::set_markers_pablo(markers, pmesh);
  }

private:
  const ConfigMap& configMap;
  const HydroParams params;
  AMRmesh& pmesh;
  Timers& timers;
  real_t error_min, error_max;
  uint32_t nbOctsPerGroup;
  
};

} // namespace muscl_block 
} // namespace dyablo 