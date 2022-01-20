#pragma once

#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "muscl_block/foreach_cell/ForeachCell.h"

#include "shared/HydroState.h"
#include "muscl_block/SolverHydroMusclBlock.h"

namespace dyablo{
namespace muscl_block{

/** 
 * Helper to implement InitialConditions with analytical formula
 * User implements the AnalyticalFormula class that implements the AnalyticalFormula_base interface
 * 
 * And the init() method :
 * 1) initializes the AMR tree by calling successively need_refine for each levels between level_min and level_max
 * 2) Fills the final AMR mesh using value()
 **/
template< typename AnalyticalFormula >
class InitialConditions_analytical : public InitialConditions{ 
    static_assert( std::is_base_of< AnalyticalFormula_base, AnalyticalFormula >::value, 
                  "AnalyticalFormula must implement AnalyticalFormula_base" );

    AnalyticalFormula analytical_formula;

    struct Data{
        AMRmesh& pmesh;
        FieldManager fieldMgr;
        uint32_t nbOctsPerGroup;  
        uint32_t bx, by, bz;
        int level_min, level_max;
        real_t xmin, ymin, zmin;
        real_t xmax, ymax, zmax;
    } data;
public:
  InitialConditions_analytical(
        ConfigMap& configMap, 
        AMRmesh& pmesh,
        FieldManager fieldMgr,
        uint32_t nbOctsPerGroup,  
        uint32_t bx, uint32_t by, uint32_t bz,  
        Timers& timers )
  : analytical_formula(configMap),
    data({
      pmesh, fieldMgr, nbOctsPerGroup, 
      bx, by, bz,
      pmesh.get_level_min(), pmesh.get_level_max(),
      configMap.getValue<real_t>("mesh", "xmin", 0.0), 
      configMap.getValue<real_t>("mesh", "ymin", 0.0), 
      configMap.getValue<real_t>("mesh", "zmin", 0.0),
      configMap.getValue<real_t>("mesh", "xmax", 1.0),
      configMap.getValue<real_t>("mesh", "ymax", 1.0),
      configMap.getValue<real_t>("mesh", "zmax", 1.0)
    })
  {}

  void init( ForeachCell::CellArray_global_ghosted& U )
  {
    AMRmesh&                 pmesh     = data.pmesh;
    FieldManager             fieldMgr  = data.fieldMgr;

    int ndim = pmesh.getDim();
    int level_min = data.level_min;
    int level_max = data.level_max;
    int bx = data.bx;
    int by = data.by;
    int bz = (ndim == 3) ? data.bz : 1;
    int xmin = data.xmin;
    int ymin = data.ymin;
    int zmin = data.zmin;
    int xmax = data.xmax;
    int ymax = data.ymax;
    int zmax = data.zmax;
    int nbOctsPerGroup = data.nbOctsPerGroup; // arbitrary, not really useful    

    // Refine to level_min
    for (uint8_t level=0; level<level_min; ++level)
        pmesh.adaptGlobalRefine();
    // Distribute uniform mesh at level_min
    pmesh.loadBalance();

    AnalyticalFormula& analytical_formula = this->analytical_formula;

    // Refine until level_max using analytical markers
    for (uint8_t level=level_min; level<level_max; ++level)
    {
        const LightOctree& lmesh = pmesh.getLightOctree();
        uint32_t nbOcts = lmesh.getNumOctants();

        ForeachCell foreach_cell(
            ndim,
            lmesh, 
            bx, by, bz, 
            xmin, ymin, zmin,
            xmax, ymax, zmax,
            nbOcts
        );

        ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
        U  = foreach_cell.allocate_ghosted_array( "U" , pmesh, fieldMgr );

        Kokkos::View<bool*> markers( "InitialConditions_analytical::markers", nbOcts );
        // Apply refine condition given by AnalyticalFormula::need_refine for each cell
        // An octant needs to be refined if at least one cell verifies the condition
        foreach_cell.foreach_cell( "InitialConditions_analytical::mark_octants", U,
                    CELL_LAMBDA( const ForeachCell::CellIndex& iCell_U )
        {
            ForeachCell::CellMetaData::pos_t c = cellmetadata.getCellCenter(iCell_U);
            ForeachCell::CellMetaData::pos_t s = cellmetadata.getCellSize(iCell_U);

            bool need_refine = analytical_formula.need_refine( c[IX], c[IY], c[IZ], s[IX], s[IY], s[IZ] );
            
            Kokkos::atomic_or( &markers(iCell_U.iOct.iOct), need_refine );
        });

        // Copy computed markers to pmesh
        auto markers_host = Kokkos::create_mirror_view(markers);
        Kokkos::deep_copy( markers_host, markers );
        Kokkos::parallel_for( "InitialConditions_analytical::copy_octants",
            Kokkos::RangePolicy<Kokkos::OpenMP>(Kokkos::OpenMP(), 0, nbOcts),
            [&](uint32_t iOct)
        {
            if( markers_host(iOct) )
                pmesh.setMarker(iOct, 1);
        });

        // Refine the mesh according to markers
        pmesh.adapt();
        // Load balance at each level to avoid excessive inbalance
        pmesh.loadBalance();
    }

    // Reallocate and fill U
    {
        const LightOctree& lmesh = pmesh.getLightOctree();

        ForeachCell foreach_cell(
            ndim,
            lmesh, 
            bx, by, bz, 
            xmin, ymin, zmin,
            xmax, ymax, zmax,
            nbOctsPerGroup
        );

        // TODO wrap reallocation in U.reallocate(pmesh)?
        U  = foreach_cell.allocate_ghosted_array( "U" , pmesh, fieldMgr );

        ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

        foreach_cell.foreach_cell( "InitialConditions_analytical::fill_U", U,
                    CELL_LAMBDA( const ForeachCell::CellIndex& iCell_U )
        {
            ForeachCell::CellMetaData::pos_t c = cellmetadata.getCellCenter(iCell_U);
            ForeachCell::CellMetaData::pos_t s = cellmetadata.getCellSize(iCell_U);

            HydroState3d val = analytical_formula.value( c[IX], c[IY], c[IZ], s[IX], s[IY], s[IZ] );
            U.at(iCell_U, ID) = val[ID];
            U.at(iCell_U, IP) = val[IP];
            U.at(iCell_U, IU) = val[IU];
            U.at(iCell_U, IV) = val[IV];
            if( ndim == 3 )
                U.at(iCell_U, IW) = val[IW];
        });
    }
  }  
}; 

} // namespace muscl_block

} // namespace dyablo