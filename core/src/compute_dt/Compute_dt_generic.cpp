#include "Compute_dt_base.h"

#include "utils_hydro.h"

namespace dyablo {


class Compute_dt_generic : public Compute_dt
{
public:
  Compute_dt_generic(   ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
  : foreach_cell(foreach_cell),
    cfl( configMap.getValue<real_t>("hydro", "cfl", 0.5) ),
    gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
    smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
    smallc( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
    smallp( smallc*smallc/gamma0 )
  {}

  double compute_dt( const ForeachCell::CellArray_global_ghosted& U)
  {
    int ndim = foreach_cell.getDim();
    real_t gamma0 = this->gamma0;
    real_t smallr = this->smallr;
    real_t smallp = this->smallp;

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt", U,
    CELL_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];
      
      HydroState3d uLoc;
      uLoc[ID] = U.at(iCell,ID);
      uLoc[IP] = U.at(iCell,IP);
      uLoc[IU] = U.at(iCell,IU);
      uLoc[IV] = U.at(iCell,IV);
      uLoc[IW] = (ndim==2)? 0 : U.at(iCell, IW);

      real_t c;
      HydroState3d qLoc;
      computePrimitives(uLoc, &c, qLoc, gamma0, smallr, smallp);

      real_t vx = c + qLoc[IU];
      real_t vy = c + qLoc[IV];
      real_t vz =  (ndim==2)? 0 : c + qLoc[IW];

      inv_dt_update = FMAX( inv_dt_update, vx/dx + vy/dy + vz/dz );

    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt = cfl / inv_dt;
    assert(dt>0);
    return dt;
  }

private:
  ForeachCell& foreach_cell;

  real_t cfl;
  real_t gamma0, smallr, smallc, smallp;  
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_generic, "Compute_dt_generic" );
