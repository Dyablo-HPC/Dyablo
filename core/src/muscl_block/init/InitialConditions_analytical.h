#pragma once

#include "InitialConditions_base.h"

#include "muscl_block/foreach_cell/AMRBlockForeachCell_group.h"
//#include "muscl_block/foreach_cell/AMRBlockForeachCell_scratch.h"

#include "shared/HydroState.h"
#include "muscl_block/SolverHydroMusclBlock.h"

namespace dyablo{
namespace muscl_block{

/** 
 * Helper to implement InitialConditions with analytical formula
 * User implements the AnalyticalFormula class that provides :
 * 1) `bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz );`
 *  determines if the cell at position {x,y,z} with size {dx,dy,dz} needs to be refined
 * 2) `HydroState3d value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz );`
 *  the final value hydro state for the cell at position {x,y,z} with size {dx,dy,dz}
 * 
 * And the init() method :
 * 1) initializes the AMR tree by calling successively need_refine for each levels between level_min and level_max
 * 2) Fills the final AMR mesh using value()
 **/
template< typename AnalyticalFormula >
class InitialConditions_analytical : public InitialConditions{ 
public:
  void init(SolverHydroMusclBlock* psolver) 
  {
    using ForeachCell = AMRBlockForeachCell_group;
    //using ForeachCell = AMRBlockForeachCell_scratch;

    std::shared_ptr<AMRmesh> amr_mesh  = psolver->amr_mesh;
    ConfigMap&               configMap = psolver->configMap;
    HydroParams&             params    = psolver->params;
    id2index_t fm                      = psolver->fieldMgr.get_id2index();

    AnalyticalFormula analytical_formula(configMap, params);

    int ndim = amr_mesh->getDim();
    int level_min = params.level_min;
    int level_max = params.level_max;
    int bx = psolver->bx;
    int by = psolver->by;
    int bz = (ndim == 3) ? psolver->bz : 1;
    int xmin = params.xmin;
    int ymin = params.ymin;
    int zmin = params.zmin;
    int xmax = params.xmax;
    int ymax = params.ymax;
    int zmax = params.zmax;
    int nbOctsPerGroup = 1024; // arbitrary, not really useful    

    // Refine to level_min
    for (uint8_t level=0; level<level_min; ++level)
        amr_mesh->adaptGlobalRefine();
    // Distribute uniform mesh at level_min
    amr_mesh->loadBalance();

    // Refine until level_max using analytical markers
    for (uint8_t level=level_min; level<level_max; ++level)
    {
        const LightOctree& lmesh = amr_mesh->getLightOctree();
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
        psolver->resize_solver_data();
        ForeachCell::CellArray_global U = foreach_cell.get_global_array(psolver->U, 0, 0, 0, fm);

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

        // Copy computed markers to amr_mesh
        auto markers_host = Kokkos::create_mirror_view(markers);
        Kokkos::deep_copy( markers_host, markers );
        Kokkos::parallel_for( "InitialConditions_analytical::copy_octants",
            Kokkos::RangePolicy<Kokkos::OpenMP>(Kokkos::OpenMP(), 0, nbOcts),
            [&](uint32_t iOct)
        {
            if( markers_host(iOct) )
                amr_mesh->setMarker(iOct, 1);
        });

        // Refine the mesh according to markers
        amr_mesh->adapt();
        // Load balance at each level to avoid excessive inbalance
        amr_mesh->loadBalance();
    }

    // Reallocate user data to fit amr_mesh
    psolver->resize_solver_data();
    // Fill U
    {
        const LightOctree& lmesh = amr_mesh->getLightOctree();

        ForeachCell foreach_cell(
            ndim,
            lmesh, 
            bx, by, bz, 
            xmin, ymin, zmin,
            xmax, ymax, zmax,
            nbOctsPerGroup
        );

        ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
        ForeachCell::CellArray_global U = foreach_cell.get_global_array(psolver->U, 0, 0, 0, fm);

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