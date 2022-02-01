#include <memory>

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/amr/LightOctree.h"
#include "muscl_block/update/MusclBlockUpdate_base.h"

#include "muscl_block/foreach_cell/ForeachCell.h"
#include "shared/utils_hydro.h"
#include "shared/RiemannSolvers.h"
#include "utils/config/ConfigMap.h"

#ifdef __CUDA_ARCH__
#include "math_constants.h"
#endif

class Timers;
class ConfigMap;

namespace dyablo {
namespace muscl_block {

namespace{
struct BoundaryConditions {
  BoundaryConditionType boundary_type_xmin, boundary_type_ymin, boundary_type_zmin;
  BoundaryConditionType boundary_type_xmax, boundary_type_ymax, boundary_type_zmax;
};
}

using AMRBlockForeachCell = AMRBlockForeachCell_group;
//using AMRBlockForeachCell = AMRBlockForeachCell_scratch;

class MusclBlockUpdate_muscl_oneneighbor : public MusclBlockUpdate{
public: 
  MusclBlockUpdate_muscl_oneneighbor(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : foreach_cell(foreach_cell),
      riemann_params(configMap),
      xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ),
      ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ),
      zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ),
      xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
      ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
      zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
      boundary_conditions {
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING)
      },
      timers(timers)
  {}

  void update(  const ForeachCell::CellArray_global_ghosted& Uin,
                const ForeachCell::CellArray_global_ghosted& Uout,
                real_t dt)
  {
    int ndim = foreach_cell.getDim();
    if(ndim==2)
      update_aux<2>(Uin,Uout, dt);
    else if(ndim==3)
      update_aux<3>(Uin,Uout, dt);
    else assert(false);
  }

  template< int ndim >
  void update_aux(  const ForeachCell::CellArray_global_ghosted& Uin,
                    const ForeachCell::CellArray_global_ghosted& Uout,
                    real_t dt);
  private:
    ForeachCell& foreach_cell;
    RiemannParams riemann_params;  
    real_t xmin, ymin, zmin;
    real_t xmax, ymax, zmax; 

    BoundaryConditions boundary_conditions;
    
    Timers& timers;
};

namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using CellIndex = ForeachCell::CellIndex;

template< int ndim, typename Array_t >
KOKKOS_INLINE_FUNCTION
HydroState3d getHydroState( const Array_t& U, const CellIndex& iCell )
{
  HydroState3d res;
  res[ID] = U.at(iCell, ID);
  res[IP] = U.at(iCell, IP);
  res[IU] = U.at(iCell, IU);
  res[IV] = U.at(iCell, IV);
  res[IW] = (ndim==3) ? U.at(iCell, IW) : 0;
  return res;
}

template< int ndim, typename Array_t >
KOKKOS_INLINE_FUNCTION
void setHydroState( const Array_t& U, const CellIndex& iCell, const HydroState3d& state )
{
  U.at(iCell, ID) = state[ID];
  U.at(iCell, IP) = state[IP];
  U.at(iCell, IU) = state[IU];
  U.at(iCell, IV) = state[IV];
  if(ndim==3)
    U.at(iCell, IW) = state[IW];
}

//Copied from MusclBlockUpdate_generic
template< int ndim >
KOKKOS_INLINE_FUNCTION
void compute_primitives(
  const RiemannParams& params, const GhostedArray& Ugroup, 
  const CellIndex& iCell_Ugroup, const GhostedArray& Qgroup)
{
  HydroState3d uLoc = getHydroState<ndim>( Ugroup, iCell_Ugroup );
      
  // get primitive variables in current cell
  HydroState3d qLoc{};
  real_t c = 0.0;
  if(ndim==3)
    computePrimitives(uLoc, &c, qLoc, params.gamma0, params.smallr, params.smallp);
  else
  {
    auto copy_state = [](auto& to, const auto& from){
      to[ID] = from[ID];
      to[IP] = from[IP];
      to[IU] = from[IU];
      to[IV] = from[IV];
    };
    HydroState2d uLoc_2d, qLoc_2d;
    copy_state(uLoc_2d, uLoc);
    computePrimitives(uLoc_2d, &c, qLoc_2d, params.gamma0, params.smallr, params.smallp);
    copy_state(qLoc, qLoc_2d);
  }

  setHydroState<ndim>( Qgroup, iCell_Ugroup, qLoc );
}

KOKKOS_INLINE_FUNCTION
real_t big_real()
{
  return 1e99;
}

/**
 * Compute limited slopes and store result in Slopes_xyz
 * @tparam ndim 2d or 3d
 * @param[in] Q Array containing primitive variables
 * @param[in] iCell_U is current cell index
 * @param pos_cell cell center position
 * @param cell_size size of the cell
 * @param Slope_xyz limited slopes for every cell (in the xyz direction)
 **/
template< int ndim >
KOKKOS_INLINE_FUNCTION
void compute_limited_slopes(const GhostedArray& Q, const CellIndex& iCell_U, 
                            ForeachCell::CellMetaData::pos_t pos_cell, ForeachCell::CellMetaData::pos_t cell_size,
                            const GhostedArray& Slope_x, const GhostedArray& Slope_y, const GhostedArray& Slope_z)
{
    constexpr VarIndex vars[] = {ID,IP,IU,IV,IW};

    for( int i=0; i< ( ndim==2 ? 4 : 5 ); i++ )
    {
        VarIndex ivar = vars[i];
        Kokkos::Array<real_t,3> grad {big_real(), big_real(), big_real()};
        
        /// Compute gradient and apply slope limiter for neighbor 'iCell_neighbor' at position 'pos_neighbor' and in direction 'dir'
        auto update_minmod =[&]( const ComponentIndex3D& dir, const CellIndex& iCell_neighbor, real_t pos_n )
        {
          // default returned value (limited gradient didn't change)
          real_t old_value = grad[dir];
          real_t new_value = old_value;

          // compute distance along direction "dir" between current and
          // neighbor cell
          real_t pos_c = pos_cell[dir];
          real_t delta_x = pos_n - pos_c;

          real_t new_grad = (Q.at(iCell_neighbor, ivar) - Q.at(iCell_U, ivar))/delta_x;

          // this first test ensure a correct initialization
          if ( old_value == big_real() )
            new_value = new_grad;
          else if (old_value * new_grad < 0)
            new_value = 0.0;
          else if ( fabs(new_grad) < fabs(old_value) )
            new_value = new_grad;

          grad[dir] = new_value;
        };

        // Compute gradient and apply slope limiter for every neighbor in direction dir (sign = Left(-1)/Right(1))
        auto process_neighbors = [&]( ComponentIndex3D dir, int sign )
        {
            CellIndex::offset_t offset{};
            offset[dir] = sign;
            CellIndex iCell_n0 = iCell_U.getNeighbor_ghost( offset, Q );

            if( iCell_n0.level_diff() == 0 ) // same size
                update_minmod(dir, iCell_n0, pos_cell[dir]+offset[dir]*cell_size[dir]);
            if( iCell_n0.level_diff() == 1 ) // bigger
                update_minmod(dir, iCell_n0, pos_cell[dir]+1.5*offset[dir]*cell_size[dir]);
            if( iCell_n0.level_diff() == -1 ) // smaller
            {
                // Iterate over adjacent neighbors
                int di_count = (offset[IX]==0)?2:1;
                int dj_count = (offset[IY]==0)?2:1;
                int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
                for( int8_t dk=0; dk<dk_count; dk++ )
                for( int8_t dj=0; dj<dj_count; dj++ )
                for( int8_t di=0; di<di_count; di++ )
                {
                    CellIndex iCell_neighbor = iCell_n0.getNeighbor_ghost({di,dj,dk}, Q);
                    update_minmod(dir, iCell_neighbor, pos_cell[dir]+0.75*offset[dir]*cell_size[dir]);
                }
            }
        };

        process_neighbors(IX, +1);
        process_neighbors(IX, -1);
        process_neighbors(IY, +1);
        process_neighbors(IY, -1);
        if(ndim==3)
        {
          process_neighbors(IZ, +1);
          process_neighbors(IZ, -1);
        }

        Slope_x.at( iCell_U, ivar ) = grad[IX];
        Slope_y.at( iCell_U, ivar ) = grad[IY];
        if(ndim==3)
          Slope_z.at( iCell_U, ivar ) = grad[IZ];
          
    }
}

/**
   * Compute Riemann fluxes and update current cell.
   *
   * @param Uin Initial fields
   * @param Uout Fields to update
   * @param Q Primitive fields
   * @param iCell_Q current cell index
   * @param Slope_xyz limited slopes for every cell (in the xyz direction)
   * @param cellmetadata cell meta data to retrieve cell size and position
   * @param dt timestep
   * @param params HydroParams configuration
   */
template< int ndim >
KOKKOS_INLINE_FUNCTION
void compute_fluxes_and_update( const GhostedArray& Uin, const GhostedArray& Uout, const GhostedArray& Q, const CellIndex& iCell_Q,
                                const GhostedArray& Slopes_x, const GhostedArray& Slopes_y, const GhostedArray& Slopes_z,
                                const ForeachCell::CellMetaData& cellmetadata, real_t dt, const RiemannParams& params,
                                const BoundaryConditions& bc)
{
  ForeachCell::CellMetaData::pos_t cell_size = cellmetadata.getCellSize(iCell_Q);
  ForeachCell::CellMetaData::pos_t pos_c = cellmetadata.getCellCenter(iCell_Q);

  HydroState3d qprim = getHydroState<ndim>( Q, iCell_Q );
  HydroState3d qcons{0}; 
  if( !params.rsst_enabled) qcons = getHydroState<ndim>( Uin, iCell_Q );
  
  /**
   * Solve riemann problem at interface between cells
   * @param qr_c primitive variables for current cell
   * @param qr_n primitive variables for neighbor cell
   * @param dir direction of interface between cells (e.g IX if neighbor is left of cell)
   * @param int sign : -1 neighbor is left of current cell, +1 neighbor is right
   * @return flux between cells
   **/
  auto riemann = [&](HydroState3d qr_c, HydroState3d qr_n, ComponentIndex3D dir, int sign)
  {
    if( dir == IY )
    {
      swap(qr_c[IU], qr_c[IV]);
      swap(qr_n[IU], qr_n[IV]);
    }
    if( dir == IZ )
    {
      swap(qr_c[IU], qr_c[IW]);
      swap(qr_n[IU], qr_n[IW]);
    }

    HydroState3d& qr_L = (sign<0)?qr_n:qr_c;
    HydroState3d& qr_R = (sign<0)?qr_c:qr_n;
    HydroState3d flux;

    riemann_hydro(qr_L,qr_R,flux,params);

    if( dir == IY )
      swap(flux[IU], flux[IV]);
    if( dir == IZ )
      swap(flux[IU], flux[IW]);

    return flux;
  };

  
  using reconstruct_offset_t = Kokkos::Array<real_t, 3>;

  /**
   * Reconstruct state form cell values and slopes at the iven position
   * 
   * @param q initial cell value
   * @param iCell_U cell position
   * @param Position on cell border where the primitive variables must be reconstructed
   * using the limited slopes.
   *
   * In 2D, offsets lies in the following square (mapping current cell)
   *
   *  (-1,1) --- (0,1) ---- (1,1) 
   *    |          |          |
   *    |          |          |
   *    |          |          |
   *  (-1,0) --- (0,0) ---- (1,0)
   *    |          |          |
   *    |          |          |
   *    |          |          |
   *  (-1,-1) ---(0,-1) --- (1,-1) 
   * @param dx_over_2 half cell size
   **/
  auto reconstruct_state = [&] ( HydroState3d q, 
                            const CellIndex& iCell_U,
                            reconstruct_offset_t offsets,
                            ForeachCell::CellMetaData::pos_t cell_size, real_t dt )
  {
    assert( ndim==3 || offsets[IZ]==0 );

    const double gamma  = params.gamma0;
    const double smallr = params.smallr;
    
    const real_t dtdx = dt/cell_size[IX];
    const real_t dtdy = dt/cell_size[IY];

    HydroState3d diff_x = getHydroState<ndim>( Slopes_x, iCell_U ) * cell_size[IX] * 0.5;
    HydroState3d diff_y = getHydroState<ndim>( Slopes_y, iCell_U ) * cell_size[IY] * 0.5;
    HydroState3d diff_z = {};

    // retrieve primitive variables in current quadrant
    real_t r = q[ID];
    real_t p = q[IP];
    real_t u = q[IU];
    real_t v = q[IV];     
    
    // retrieve variations = dx * slopes
    real_t drx = diff_x[ID];
    real_t dpx = diff_x[IP];
    real_t dux = diff_x[IU];
    real_t dvx = diff_x[IV];    

    real_t dry = diff_y[ID];
    real_t dpy = diff_y[IP];
    real_t duy = diff_y[IU];
    real_t dvy = diff_y[IV];    

    HydroState3d qs{};
    if( ndim == 3 )
    {
      const real_t dtdz = dt/cell_size[IZ];

      real_t w = q[IW];
      real_t dwx = diff_x[IW];
      real_t dwy = diff_y[IW];

      diff_z = getHydroState<ndim>( Slopes_z, iCell_U ) * cell_size[IZ] * 0.5;

      real_t drz = diff_z[ID];
      real_t dpz = diff_z[IP];
      real_t duz = diff_z[IU];
      real_t dvz = diff_z[IV];
      real_t dwz = diff_z[IW];

      qs[ID] = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy + (-w * drz - dwz * r) * dtdz;
      qs[IU] = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy + (-w * duz) * dtdz;
      qs[IV] = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy + (-w * dvz) * dtdz;
      qs[IW] = w + (-u * dwx) * dtdx + (-v * dwy) * dtdy + (-w * dwz - dpz / r) * dtdz;
      qs[IP] = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy + (-w * dpz - dwz * gamma * p) * dtdz;
    }
    else
    {
      qs[ID] = r + (-u * drx - dux * r) * dtdx + (-v * dry - dvy * r) * dtdy;
      qs[IU] = u + (-u * dux - dpx / r) * dtdx + (-v * duy) * dtdy;
      qs[IV] = v + (-u * dvx) * dtdx + (-v * dvy - dpy / r) * dtdy;
      qs[IP] = p + (-u * dpx - dux * gamma * p) * dtdx + (-v * dpy - dvy * gamma * p) * dtdy;
    }

    // reconstruct state on interface
    HydroState3d qr = qs + diff_x * offsets[IX] + diff_y * offsets[IY] + diff_z * offsets[IZ];
    qr[ID] = fmax(smallr, qr[ID]);

    return qr;    
  };

  // Process every neighbor in direction dir (sign = Left(-1)/Right(1))
  auto process_dir = [&]( ComponentIndex3D dir, int sign )
  {
    CellIndex::offset_t offset{};
    offset[dir] = sign;
    CellIndex iCell_n0 = iCell_Q.getNeighbor_ghost( offset, Q );

    HydroState3d qr_c, qr_n;

    if( iCell_n0.is_boundary() )
    {
      qr_c = qprim;
      qr_n = qprim;
      if( (bc.boundary_type_xmin == BC_REFLECTING && iCell_Q.getNeighbor_ghost( {-1,0,0}, Q ).is_boundary())
      ||  (bc.boundary_type_xmax == BC_REFLECTING && iCell_Q.getNeighbor_ghost( {+1,0,0}, Q ).is_boundary()) )
        qr_n[IU] = -qr_n[IU];
      if( (bc.boundary_type_ymin == BC_REFLECTING && iCell_Q.getNeighbor_ghost( {0,-1,0}, Q ).is_boundary())
      ||  (bc.boundary_type_ymax == BC_REFLECTING && iCell_Q.getNeighbor_ghost( {0,+1,0}, Q ).is_boundary()) )
        qr_n[IV] = -qr_n[IV];
      if( (ndim==3) &&
        ( (bc.boundary_type_zmin == BC_REFLECTING && iCell_Q.getNeighbor_ghost( {0,0,-1}, Q ).is_boundary())
      ||  (bc.boundary_type_zmax == BC_REFLECTING && iCell_Q.getNeighbor_ghost( {0,0,+1}, Q ).is_boundary()) ) )
        qr_n[IW] = -qr_n[IW];

      HydroState3d flux = riemann(qr_c, qr_n, dir, sign);
      // +- dS / dV 
      real_t scale = -sign * dt / cell_size[dir];      
      qcons += flux*scale;     
    }
    else
    {
       ForeachCell::CellMetaData::pos_t cell_size_n = cellmetadata.getCellSize(iCell_n0);

      if( iCell_n0.level_diff() >= 0 ) // Only one cell
      {
        // 0. retrieve primitive variables in neighbor cell
        HydroState3d qprim_n = getHydroState<ndim>( Q, iCell_n0 );

        // 1. reconstruct primitive variables on both sides of current interface (iface)

        // current cell reconstruction  (primitive variables)
        // Position at which state is reconstructed on current cell border ([-1,1])
        // In this case, neighbor is always bigger or same size : center of face is used
        const reconstruct_offset_t offsets_c{
          (real_t)offset[IX], 
          (real_t)offset[IY], 
          (real_t)offset[IZ] 
        };
        HydroState3d qr_c = reconstruct_state(qprim, iCell_Q, offsets_c, cell_size, dt);

        // neighbor cell reconstruction (primitive variables)
        // Position at which state is reconstructed on neighbor cell border ([-1,1])
        // In this case, current cell can be smaller or same size
        //  Center of face is used if same size:
        reconstruct_offset_t offsets_n{
          (real_t)-offset[IX], 
          (real_t)-offset[IY], 
          (real_t)-offset[IZ] 
        };
        //  If "current" is smaller than "neighbor", center of "current" cell is used
        if( iCell_n0.level_diff() > 0 )
        {
          ForeachCell::CellMetaData::pos_t pos_n = cellmetadata.getCellCenter(iCell_n0);
          //We need to determine which quadrant of "neighbor"'s face is "current"'s face
          for( int facedir = 0; facedir < ndim; facedir++ )
          if( facedir != dir ) // Foreach direction inside face (orthogonal to dir)
          {
            offsets_n[facedir] += (pos_c[facedir] > pos_n[facedir])? 0.5 : -0.5;
          }
        }
        HydroState3d qr_n = reconstruct_state(qprim_n, iCell_n0, offsets_n, cell_size_n, dt);

        HydroState3d flux = riemann(qr_c, qr_n, dir, sign);
        // +- dS / dV 
        real_t scale = -sign * dt / cell_size[dir] ;      
        qcons += flux*scale; 
      }
      else // ( iCell_n0.level_diff() == -1 ) // Multiple smaller neighbors
      {
        // Accumulate fluxes from neighbors of initial cell
        int di_count = (offset[IX]==0)?2:1;
        int dj_count = (offset[IY]==0)?2:1;
        int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
        for( int8_t dk=0; dk<dk_count; dk++ )
        for( int8_t dj=0; dj<dj_count; dj++ )
        for( int8_t di=0; di<di_count; di++ )
        {
            CellIndex iCell_n = iCell_n0.getNeighbor_ghost({di,dj,dk}, Q);
            // 0. retrieve primitive variables in neighbor cell
            HydroState3d qprim_n = getHydroState<ndim>( Q, iCell_n );

            // 1. reconstruct primitive variables on both sides of current interface (iface)

            // current cell reconstruction  (primitive variables)
            // Position at which state is reconstructed on current cell border ([-1,1])
            // In this case, neighbor is always smaller : center of neighbor face is used
            // Relative position of neighbor can be computed from {di,dj,dk}
            const reconstruct_offset_t offsets_c{
              (offset[IX] != 0) ? (real_t)offset[IX] : di-0.5, 
              (offset[IY] != 0) ? (real_t)offset[IY] : dj-0.5, 
              (ndim==2 || offset[IZ] != 0) ? (real_t)offset[IZ] : dk-0.5
            };
            HydroState3d qr_c = reconstruct_state(qprim, iCell_Q, offsets_c, cell_size, dt);

            // neighbor cell reconstruction (primitive variables)
            // Position at which state is reconstructed on neighbor cell border ([-1,1])
            // In this case, current cell is bigger than neighbor, center of neighbor is used
            const reconstruct_offset_t offsets_n{
              (real_t)-offset[IX], 
              (real_t)-offset[IY], 
              (real_t)-offset[IZ] 
            };            
            HydroState3d qr_n = reconstruct_state(qprim_n, iCell_n, offsets_n, cell_size_n, dt);
            HydroState3d flux = riemann(qr_c, qr_n, dir, sign);
            // +- dS / dV 
            int nneigh = (ndim-1)*2;
            real_t scale = -sign * dt /  (cell_size[dir] * nneigh)  ;      
            qcons += flux*scale; 
        }
      }
    }
  };

  process_dir(IX, +1);
  process_dir(IX, -1);
  process_dir(IY, +1);
  process_dir(IY, -1);
  if(ndim==3)
  {
    process_dir(IZ, +1);
    process_dir(IZ, -1);
  }

  setHydroState<ndim>(Uout, iCell_Q, qcons);
}

} // namespace

template< int ndim >
void MusclBlockUpdate_muscl_oneneighbor::update_aux( 
  const ForeachCell::CellArray_global_ghosted& Uin,
  const ForeachCell::CellArray_global_ghosted& Uout,
  real_t dt)
{
    using GhostedArray = ForeachCell::CellArray_global_ghosted;
    const RiemannParams& riemann_params = this->riemann_params;
    const BoundaryConditions& boundary_conditions = this->boundary_conditions;
    ForeachCell& foreach_cell = this->foreach_cell;
    GhostCommunicator ghost_comm(std::shared_ptr<AMRmesh>(&foreach_cell.get_amr_mesh(), [](AMRmesh*){}));

    timers.get("MusclBlockUpdate_muscl_oneneighbor").start();

    std::set<VarIndex> enabled_fields = {ID,IP,IU,IV};
    if( ndim == 3) enabled_fields.insert(IW);
    FieldManager field_manager( enabled_fields );
    
    GhostedArray Q = foreach_cell.allocate_ghosted_array( "Q", field_manager );

    // Fill Q with primitive variables
    foreach_cell.foreach_cell("MusclBlockUpdate_muscl_oneneighbor::convertToPrimitives", Q, CELL_LAMBDA(const CellIndex& iCell_Q)
    { 
        compute_primitives<ndim>(riemann_params, Uin, iCell_Q, Q);
    });
    // Primitive variables of ghost cells are needed to compute slopes
    Q.exchange_ghosts(ghost_comm);

    // Create arrays to store slopes
    GhostedArray Slopes_x = foreach_cell.allocate_ghosted_array( "Slopes_x", field_manager );
    GhostedArray Slopes_y = foreach_cell.allocate_ghosted_array( "Slopes_Y", field_manager );
    GhostedArray Slopes_z;
    if(ndim == 3)
      Slopes_z = foreach_cell.allocate_ghosted_array( "Slopes_z", field_manager );

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Fill slope arrays
    foreach_cell.foreach_cell("MusclBlockUpdate_muscl_oneneighbor::reconstruct_gradients", Q, CELL_LAMBDA(const CellIndex& iCell_Q)
    { 
        compute_limited_slopes<ndim>(Q, iCell_Q, cellmetadata.getCellCenter(iCell_Q), cellmetadata.getCellSize(iCell_Q), Slopes_x, Slopes_y, Slopes_z);
    });
    // Slopes of ghost cells are needed to compute flux
    Slopes_x.exchange_ghosts(ghost_comm);
    Slopes_y.exchange_ghosts(ghost_comm);
    if(ndim == 3)
      Slopes_z.exchange_ghosts(ghost_comm);

    // Compute flux and update Uout
    foreach_cell.foreach_cell("MusclBlockUpdate_muscl_oneneighbor::flux_and_update", Q, CELL_LAMBDA(const CellIndex& iCell_Q)
    { 
        compute_fluxes_and_update<ndim>(  Uin, Uout, Q, iCell_Q, 
                                          Slopes_x, Slopes_y, Slopes_z,
                                          cellmetadata, dt, riemann_params, boundary_conditions );
    });

    timers.get("MusclBlockUpdate_muscl_oneneighbor").stop();
}

} //namespace muscl_block
} //namespace dyablo 

FACTORY_REGISTER( dyablo::muscl_block::MusclBlockUpdateFactory , dyablo::muscl_block::MusclBlockUpdate_muscl_oneneighbor, "MusclBlockUpdate_muscl_oneneighbor")