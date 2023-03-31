#pragma once

#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"

#include "states/State_forward.h"

namespace dyablo{


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
        ForeachCell& foreach_cell;
        int level_min, level_max;
    } data;
public:
  InitialConditions_analytical(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : analytical_formula(configMap),
    data({
      foreach_cell,
      foreach_cell.get_amr_mesh().get_level_min(),
      foreach_cell.get_amr_mesh().get_level_max()
    })
  {}

  void init( UserData& U )
  {
    ForeachCell& foreach_cell = data.foreach_cell;
    AMRmesh& pmesh     = foreach_cell.get_amr_mesh();

    uint32_t ndim = pmesh.getDim();
    int level_min = data.level_min;
    int level_max = data.level_max;

    AnalyticalFormula& analytical_formula = this->analytical_formula;

    // Refine until level_max using analytical markers
    for (uint8_t level=level_min; level<level_max; ++level)
    {
        const LightOctree& lmesh = pmesh.getLightOctree();
        uint32_t nbOcts = lmesh.getNumOctants();

        ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
        auto Ushape = foreach_cell.allocate_ghosted_array("dummy", FieldManager(1));
        Kokkos::View<int*> markers( "InitialConditions_analytical::markers", nbOcts );
        // Apply refine condition given by AnalyticalFormula::need_refine for each cell
        // An octant needs to be refined if at least one cell verifies the condition
        foreach_cell.foreach_cell( "InitialConditions_analytical::mark_octants", Ushape,
                    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
        {
            ForeachCell::CellMetaData::pos_t c = cellmetadata.getCellCenter(iCell_U);
            ForeachCell::CellMetaData::pos_t s = cellmetadata.getCellSize(iCell_U);

            int need_refine = analytical_formula.need_refine( c[IX], c[IY], c[IZ], s[IX], s[IY], s[IZ] );
            
            Kokkos::atomic_fetch_or( &markers(iCell_U.iOct.iOct), need_refine );
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
        using State_t = decltype( std::declval<AnalyticalFormula>().value( 0,0,0, 0,0,0 ) );

        std::set<std::string> fields;
        {
            for( const auto& field_info : State_t::getFieldsInfo() )
                fields.insert( field_info.name );
        }
        U.new_fields( fields );

        ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

        const UserData::FieldAccessor Uout = U.getAccessor( State_t::getFieldsInfo() );

        foreach_cell.foreach_cell( "InitialConditions_analytical::fill_U", U.getShape(),
                    KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell_U )
        {
            ForeachCell::CellMetaData::pos_t c = cellmetadata.getCellCenter(iCell_U);
            ForeachCell::CellMetaData::pos_t s = cellmetadata.getCellSize(iCell_U);

            auto val = analytical_formula.value( c[IX], c[IY], c[IZ], s[IX], s[IY], s[IZ] );
            if (ndim == 2)
                setConservativeState<2>(Uout, iCell_U, val);
            else
                setConservativeState<3>(Uout, iCell_U, val);
        });
    }
  }  
}; 



} // namespace dyablo