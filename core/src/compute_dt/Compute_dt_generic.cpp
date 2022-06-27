#include "Compute_dt_base.h"

#include "utils_hydro.h"

#include "states/State_hydro.h"

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
    smallp( smallc*smallc/gamma0 ),
    has_mhd( configMap.getValue<std::string>("hydro", "update", "HydroUpdate_hancock").find("MHD") != std::string::npos )
  {}

  double compute_dt( const ForeachCell::CellArray_global_ghosted& U )
  {
    int ndim = foreach_cell.getDim();
    real_t gamma0 = this->gamma0;
    bool has_mhd = this->has_mhd;

    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    real_t inv_dt;
    foreach_cell.reduce_cell( "compute_dt", U,
    CELL_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& inv_dt_update )
    {
      auto cell_size = cells.getCellSize(iCell);
      real_t dx = cell_size[IX];
      real_t dy = cell_size[IY];
      real_t dz = cell_size[IZ];
      
      ConsHydroState uLoc;
      if (ndim == 2)
        getConservativeState<2>(U, iCell, uLoc);
      else
        getConservativeState<3>(U, iCell, uLoc);
      
      PrimHydroState qLoc = consToPrim<3>(uLoc, gamma0);
      const real_t cs = sqrt(qLoc.p * gamma0 / qLoc.rho);

      real_t vx = cs + qLoc.u;
      real_t vy = cs + qLoc.v;
      real_t vz = (ndim==2)? 0 : cs + qLoc.w;

      inv_dt_update = FMAX( inv_dt_update, vx/dx + vy/dy + vz/dz );

      // TODO : Find a BETTER way to do this !
      // In fine, compute_dt should be templated by the type of State
      // and calculations of inv_dt should be held in a separate State function
      if (has_mhd) {
        const real_t Bx = U.at(iCell,IBX);
        const real_t By = U.at(iCell,IBY);
        const real_t Bz = U.at(iCell,IBZ);
        const real_t gr = cs*cs*qLoc.rho;
        const real_t Bt2 [] = {By*By+Bz*Bz,
                               Bx*Bx+Bz*Bz,
                               Bx*Bx+By*By};
        const real_t B2 = Bx*Bx + By*By + Bz*Bz;
        const real_t cf1 = gr-B2;
        const real_t V [] = {qLoc.u, qLoc.v, qLoc.w};
        const real_t D [] = {dx, dy, dz};

        for (int i=0; i < ndim; ++i) {
          const real_t cf2 = gr + B2 + sqrt(cf1*cf1 + 4.0*gr*Bt2[i]);
          const real_t cf = sqrt(0.5 * cf2 / qLoc.rho);

          const real_t cmax = FMAX(FABS(V[i] - cf), FABS(V[i] + cf));
          inv_dt_update = FMAX(inv_dt_update, cmax/D[i]);
        }
     }

    }, Kokkos::Max<real_t>(inv_dt) );

    real_t dt = cfl / inv_dt;
    assert(dt>0);
    return dt;
  }

private:
  ForeachCell& foreach_cell;

  real_t cfl;
  real_t gamma0, smallr, smallc, smallp;  
  bool has_mhd;
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::Compute_dtFactory, dyablo::Compute_dt_generic, "Compute_dt_generic" );
