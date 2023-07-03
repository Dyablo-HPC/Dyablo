#pragma once

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "gravity/GravitySolver_base.h"

namespace dyablo {

/**
 * Interface to implement for AnalyticalFormula_t template parameter
 * for GravitySolver_analytical
 **/
struct AnalyticalFormula_gravity
{
public:
  struct GravityField_t{
    real_t gx, gy, gz;
  };

  /// Return gravity force for cell at position x,y,z and size dx, dy, dz
  KOKKOS_INLINE_FUNCTION
  GravityField_t value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const;
};

/**
 * Helper to write analytical force fields
 * @tparam AnalyticalFormula_t implementation of AnalyticalFormula_gravity to be applied to each cell
 **/
template< typename AnalyticalFormula_t >
class GravitySolver_analytical : public GravitySolver{
private:
  ForeachCell& foreach_cell;
  Timers& timers;
  AnalyticalFormula_t analytical_formula;
public: 
  GravitySolver_analytical(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
  :  foreach_cell(foreach_cell),
     timers(timers),
     analytical_formula(configMap)
  { 
    GravityType gtype = configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_CST_FIELD);
    
    DYABLO_ASSERT_HOST_RELEASE( gtype == GRAVITY_CST_FIELD, "GravitySolver_analytical must have gravity_type=constant_field" )
  }

  ~GravitySolver_analytical(){}

  void update_gravity_field( UserData& U, ScalarSimulationData& scalar_data )
  {
    AnalyticalFormula_t& analytical_formula = this->analytical_formula;
    ForeachCell& foreach_cell = this->foreach_cell;
    uint8_t ndim = foreach_cell.getDim();

    timers.get("GravitySolver_analytical").start();

    enum VarIndex_gravity {IGX, IGY, IGZ};

    UserData::FieldAccessor Uout = U.getAccessor({{"gx", IGX}, {"gy", IGY}, {"gz", IGZ}});
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    foreach_cell.foreach_cell( "GravitySolver_analytical::update_gravity_field", Uout.getShape(), 
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout)
    {
      auto c = cellmetadata.getCellCenter(iCell_Uout);
      auto s = cellmetadata.getCellSize(iCell_Uout);
      auto g = analytical_formula.value( c[IX], c[IY], c[IZ], s[IX], s[IY], s[IZ] );
      Uout.at( iCell_Uout, IGX ) = g.gx;
      Uout.at( iCell_Uout, IGY ) = g.gy;
      if(ndim == 3)
        Uout.at( iCell_Uout, IGZ ) = g.gz;
    });

    timers.get("GravitySolver_analytical").stop();
  }
};

} //namespace dyablo 
