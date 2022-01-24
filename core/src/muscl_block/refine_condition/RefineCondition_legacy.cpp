#include "muscl_block/refine_condition/RefineCondition_base.h"

#include "muscl_block/legacy/MarkOctantsHydroFunctor.h"
#include "muscl_block/legacy/CopyInnerBlockCellData.h"
#include "muscl_block/legacy/CopyGhostBlockCellData.h"
#include "muscl_block/legacy/ConvertToPrimitivesHydroFunctor.h"

namespace dyablo {
namespace muscl_block {

class RefineCondition_legacy : public RefineCondition
{
public:
  RefineCondition_legacy( ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : configMap(configMap),
      pmesh(foreach_cell.get_amr_mesh()),
      timers(timers),
      error_min ( configMap.getValue<real_t>("amr", "error_min", 0.2) ),
      error_max ( configMap.getValue<real_t>("amr", "error_max", 0.8) ),
      nbOctsPerGroup(configMap.getValue<uint32_t>("amr", "nbOctsPerGroup", 64)),
      gravity_type( configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE) ),
      bxmin( configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING) ),
      bxmax( configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING) ),
      bymin( configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING) ),
      bymax( configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING) ),
      bzmin( configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING) ),
      bzmax( configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING) ),
      gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
      smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
      smallc( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
      smallp( smallc*smallc/gamma0 )
  {}

  void mark_cells( const ForeachCell::CellArray_global_ghosted& U )
  {
    int ndim = pmesh.getDim();

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
      CopyInnerBlockCellDataFunctor::apply({ndim, gravity_type}, fm,
                                        {bx,by,bz},
                                        ghostWidth,
                                        nbOcts,
                                        nbOctsPerGroup,
                                        U.U, Ugroup, 
                                        iGroup);
      CopyGhostBlockCellDataFunctor::apply(pmesh.getLightOctree(),
                                          {
                                            bxmin, bxmax,
                                            bymin, bymax,
                                            bzmin, bzmax,
                                            gravity_type
                                          },
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
      ConvertToPrimitivesHydroFunctor::apply({ndim, gamma0, smallr, smallp}, 
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
      MarkOctantsHydroFunctor::apply(pmesh.getLightOctree(), 
                                    pmesh.get_level_min(), pmesh.get_level_max(), 
                                    fm,
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
  AMRmesh& pmesh;
  Timers& timers;
  real_t error_min, error_max;
  uint32_t nbOctsPerGroup;

  GravityType gravity_type;
  BoundaryConditionType bxmin;
  BoundaryConditionType bxmax;
  BoundaryConditionType bymin;
  BoundaryConditionType bymax;
  BoundaryConditionType bzmin;
  BoundaryConditionType bzmax;
  real_t gamma0, smallr, smallc, smallp;
  
};

} // namespace muscl_block 
} // namespace dyablo 

FACTORY_REGISTER( dyablo::muscl_block::RefineConditionFactory, dyablo::muscl_block::RefineCondition_legacy, "RefineCondition_legacy" );