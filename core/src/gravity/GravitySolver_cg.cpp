#include "GravitySolver_cg.h"

#include "muscl_block/utils_block.h"
#include "utils/monitoring/Timers.h"

#include "muscl_block/foreach_cell/AMRBlockForeachCell_group.h"
#include "shared/utils_hydro.h"
#include "shared/mpi/GhostCommunicator.h"
#include <mpi.h>

namespace dyablo { 
namespace muscl_block {

struct GravitySolver_cg::Data{
  const ConfigMap& configMap;
  const HydroParams& params;  
  const LightOctree& lmesh;
  const GhostCommunicator ghost_comm;
  const id2index_t fm;
 
  uint32_t bx, by, bz; 
  
  Timers& timers;  

  uint32_t nbOctsPerGroup;
};

GravitySolver_cg::GravitySolver_cg(
  const ConfigMap& configMap,
  const HydroParams& params, 
  std::shared_ptr<AMRmesh> pmesh,
  const id2index_t& fm,
  uint32_t bx, uint32_t by, uint32_t bz,
  Timers& timers )
 : pdata(new Data
    {configMap, 
    params, 
    pmesh->getLightOctree(), 
    GhostCommunicator(pmesh),
    fm,
    bx, by, bz,
    timers
    })
{

}

GravitySolver_cg::~GravitySolver_cg()
{}

namespace{

using ForeachCell = AMRBlockForeachCell_group;
using GlobalArray = typename ForeachCell::CellArray_global;
using GhostedArray = typename ForeachCell::CellArray_global_ghosted;
using CellIndex = typename ForeachCell::CellIndex;

constexpr VarIndex CG_IP = (VarIndex)0;
constexpr VarIndex CG_IR = (VarIndex)1;
constexpr VarIndex CG_PHI = (VarIndex)2;

KOKKOS_INLINE_FUNCTION
real_t get_value(const GhostedArray& U, const CellIndex& iCell_U, VarIndex var, const CellIndex::offset_t& offset)
{
  constexpr int ndim = 3;
  if( iCell_U.is_boundary() )
  {
    return 0; 
    // TODO : When using non-zero fixed value boundary conditions, use value when computing Ax_0 but 0 when computing Ap
  }  
  
  else if( iCell_U.level_diff() >= 0 )
  {
    return U.at(iCell_U, var);
  }
  else
  {
    real_t sum = 0;
    // Accumulate values from neighbors of initial cell
    int di_count = (offset[IX]==0)?2:1;
    int dj_count = (offset[IY]==0)?2:1;
    int dk_count = (ndim==3 && offset[IZ]==0)?2:1;
    int di_start = (offset[IX]==-1)?1:0;
    int dj_start = (offset[IY]==-1)?1:0;
    int dk_start = (offset[IZ]==-1 && ndim==3)?1:0;
    for( int8_t dk=0; dk<dk_count; dk++ )
    for( int8_t dj=0; dj<dj_count; dj++ )
    for( int8_t di=0; di<di_count; di++ )
    {
        CellIndex iCell_ghost = iCell_U.getNeighbor({(int8_t)(di_start+di),(int8_t)(dj_start+dj),(int8_t)(dk_start+dk)});
        sum += U.at(iCell_ghost, var);
    }
    int nbCells = di_count*dj_count*dk_count;
    return sum/nbCells;
  } 
}

KOKKOS_INLINE_FUNCTION
real_t matprod(const GhostedArray& GCdata, const CellIndex& iCell_CGdata, VarIndex var, real_t dx, real_t dy, real_t dz, const BoundaryConditionType boundarycondition[3])
{
  constexpr int ndim = 3;

  Kokkos::Array<real_t, 3> ddir { dx, dy, dz };

  real_t res = 0;
  for( ComponentIndex3D dir : {IX,IY,IZ} )
  {
    CellIndex::offset_t off_L{}; off_L[dir] = -1;
    CellIndex::offset_t off_R{}; off_R[dir] = +1;
    CellIndex iCell_L = iCell_CGdata.getNeighbor_ghost(off_L, GCdata);
    CellIndex iCell_R = iCell_CGdata.getNeighbor_ghost(off_R, GCdata);
    real_t S = (dx*dy*dz)/ddir[dir];

    real_t hl = ddir[dir];
    real_t hr = ddir[dir];  
    if( iCell_L.level_diff()==-1 ) hl *= 0.75; 
    if( iCell_R.level_diff()==-1 ) hr *= 0.75;
    if( iCell_L.level_diff()==1 ) hl *= 1.5; 
    if( iCell_R.level_diff()==1 ) hr *= 1.5;

    real_t dvar_L = S*( GCdata.at(iCell_CGdata, var) - get_value( GCdata, iCell_L, var, off_L ) )/hl;
    real_t dvar_R = S*( get_value( GCdata, iCell_R, var, off_R ) - GCdata.at(iCell_CGdata, var) )/hr;
    
    if( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) dvar_L = 0;
    if( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) dvar_R = 0;
    
    res += (dvar_R - dvar_L);
  }
  return -res;
}

/// Right-hand term for linear solver : 4 Pi G rho
KOKKOS_INLINE_FUNCTION
real_t b(const GhostedArray& Uin, const CellIndex& iCell_Uin, real_t rho_mean, real_t gravity_constant, ForeachCell::CellMetaData::pos_t size)
{
  return -gravity_constant*(Uin.at(iCell_Uin, ID)-rho_mean)*size[IX]*size[IY]*size[IZ];
}

real_t MPI_Allreduce_scalar( real_t local_v )
{
  real_t res;
  MPI_Allreduce( &local_v, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  return res;
}

} // namespace

void GravitySolver_cg::update_gravity_field(
    DataArrayBlock U_, DataArrayBlock Ughost_,
    DataArrayBlock Uout_)
{
  const HydroParams& params = pdata->params;
  const LightOctree& lmesh = pdata->lmesh;
  uint8_t ndim = lmesh.getNdim();
  const id2index_t& fm = pdata->fm;
  uint32_t bx = pdata->bx, by = pdata->by, bz = pdata->bz;
  real_t xmin = params.xmin, ymin = params.ymin, zmin = params.zmin;
  real_t xmax = params.xmax, ymax = params.ymax, zmax = params.zmax;
  const uint32_t nbOcts = lmesh.getNumOctants();
  const uint32_t nbGhosts = lmesh.getNumGhosts();
  const uint32_t nbOctsPerGroup = nbOcts;
  const GhostCommunicator& ghost_comm = pdata->ghost_comm;
 BoundaryConditionType boundarycondition[3] = {params.boundary_type_xmin,params.boundary_type_ymin,params.boundary_type_zmin};

  real_t gravity_constant = 1;// 4 pi G
  real_t eps = 1E-3; // sqrt( mean square diff )

  ForeachCell foreach_cell(
    ndim,
    lmesh, 
    bx, by, bz,
    xmin, ymin, zmin,
    xmax, ymax, zmax,
    nbOctsPerGroup
  );
  ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

  GhostedArray Uin =  foreach_cell.get_ghosted_array(U_, Ughost_, lmesh, fm);
  GlobalArray Uout = foreach_cell.get_global_array(Uout_, 0, 0, 0, fm);

  // Setup field manager for CGdata
  FieldManager fieldManager( {CG_IP, CG_IR, CG_PHI} );
  int fm_gc_count = fieldManager.nbfields();
  id2index_t fm_cg = fieldManager.get_id2index();
  // Allocate global array
  DataArrayBlock CGdata_("CGdata", bx*by*bz, fm_gc_count, nbOcts );
  DataArrayBlock CGdata_ghosts("CGdata", bx*by*bz, fm_gc_count, nbGhosts);
  // Link global Array to GhostedArray
  GhostedArray CGdata = foreach_cell.get_ghosted_array(CGdata_, CGdata_ghosts, lmesh, fm_cg);

  pdata->timers.get("GravitySolver_cg").start();

  real_t rho_mean = 0;
  foreach_cell.reduce_cell( "Gravity_cg::rho_mean", CGdata, 
    CELL_LAMBDA(const CellIndex& iCell_CGdata, real_t& update_rhomean)
  {
    ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell_CGdata);
    real_t rhoi = Uin.at(iCell_CGdata, ID);
    update_rhomean += rhoi*size[IX]*size[IY]*size[IZ];
  }, Kokkos::Sum<real_t>(rho_mean) );
  real_t Vtot = (xmax-xmin)*(ymax-ymin)*(zmax-zmin);
  rho_mean = MPI_Allreduce_scalar(rho_mean)/Vtot;

  real_t B = 0;
  foreach_cell.reduce_cell( "Gravity_cg::B", CGdata, 
    CELL_LAMBDA(const CellIndex& iCell_CGdata, real_t& update_b)
  {
    ForeachCell::CellMetaData::pos_t size = cells.getCellSize(iCell_CGdata);
    real_t bi = b(Uin, iCell_CGdata, rho_mean, gravity_constant, size);
    update_b += bi*bi ;
  }, Kokkos::Sum<real_t>(B) );  
  B = MPI_Allreduce_scalar(B);

  // Compute initial residual
  real_t R = 0;
  foreach_cell.reduce_cell( "Gravity_cg::init_gc", CGdata, 
    CELL_LAMBDA(const CellIndex& iCell_CGdata, real_t& update_r)
  {
    // Get gravity potential from last iteration as approximate solution
    CGdata.at(iCell_CGdata, CG_PHI) = Uin.at(iCell_CGdata, IGPHI);
    auto cell_size = cells.getCellSize(iCell_CGdata);
    real_t r = b(Uin, iCell_CGdata, rho_mean, gravity_constant, cell_size)-matprod(Uin, iCell_CGdata, IGPHI, cell_size[IX], cell_size[IY], cell_size[IZ], boundarycondition);
    CGdata.at(iCell_CGdata, CG_IR) = r; // r = b-A*x0
    CGdata.at(iCell_CGdata, CG_IP) = r; // p = r
    update_r += r*r ;
  }, Kokkos::Sum<real_t>(R) );  
  R = MPI_Allreduce_scalar(R);
    
  std::cout << "GC : " << R << "...";
  // Conjugate gradient iteration
  int it=0;
  while( R/B > eps*eps )
  {
    it++;
    // Communicate ghosts for CG_IP
    // TODO : exchange only CG_IP
    CGdata.exchange_ghosts(ghost_comm);

    // Compute p'Ap for alpha
    real_t pAp = 0;
    foreach_cell.reduce_cell( "Gravity_cg::pAp", CGdata, 
      CELL_LAMBDA(const CellIndex& iCell_CGdata, real_t& pAp)
    {
      auto cell_size = cells.getCellSize(iCell_CGdata);

      real_t pi = CGdata.at(iCell_CGdata, CG_IP) ;
      real_t Api = matprod(CGdata, iCell_CGdata, CG_IP, cell_size[IX], cell_size[IY], cell_size[IZ], boundarycondition) ;
      real_t piApi = pi*Api;
      pAp += piApi;
    }, Kokkos::Sum<real_t>(pAp));
    pAp = MPI_Allreduce_scalar(pAp);

    // Update 
    // alpha <- R/p'Ap
    // phi   <- phi + alpha * p
    // r     <- r   - alpha * Ap
    // Rnext <- ||r||
    real_t Rnext = 0;
    foreach_cell.reduce_cell( "Gravity_cg::alpha", CGdata, 
      CELL_LAMBDA(const CellIndex& iCell_CGdata, real_t& Rnext)
    {
      auto cell_size = cells.getCellSize(iCell_CGdata);
      real_t alpha = R/pAp;
      CGdata.at(iCell_CGdata, CG_PHI) += alpha*CGdata.at(iCell_CGdata, CG_IP);
      real_t r = CGdata.at(iCell_CGdata, CG_IR) - alpha*matprod(CGdata, iCell_CGdata, CG_IP, cell_size[IX], cell_size[IY], cell_size[IZ], boundarycondition);
      CGdata.at(iCell_CGdata, CG_IR) = r;
      Rnext += r*r;
    }, Kokkos::Sum<real_t>(Rnext));
    Rnext = MPI_Allreduce_scalar(Rnext);

    // Update p <- r + Rnext/R * p
    foreach_cell.foreach_cell( "Gravity_cg::beta", CGdata, 
      CELL_LAMBDA(const CellIndex& iCell_CGdata)
    {
      real_t beta = Rnext/R;
      CGdata.at(iCell_CGdata, CG_IP) = CGdata.at(iCell_CGdata, CG_IR) + beta * CGdata.at(iCell_CGdata, CG_IP);
    });

    R = Rnext;
  }
  std::cout << R << " : " << it << " iter" << std::endl;
  // Communicate ghosts for CG_PHI
  // TODO : exchange only CG_PHI 
  CGdata.exchange_ghosts(ghost_comm);

  // Update force field in U from potential
  foreach_cell.foreach_cell( "Gravity_cg::construct_force_field", Uout, 
    CELL_LAMBDA(const CellIndex& iCell_Uout)
  { 
    auto size = cells.getCellSize(iCell_Uout);
    CellIndex iCell_CGdata = CGdata.convert_index(iCell_Uout);

    auto gradient = [&](ComponentIndex3D dir)
    {
      real_t phy_C = CGdata.at(iCell_CGdata, CG_PHI);
      Uout.at(iCell_Uout, IGPHI) = phy_C;
      CellIndex::offset_t off_L = {}; off_L[dir] = -1;
      CellIndex::offset_t off_R = {}; off_R[dir] = +1;
      CellIndex iCell_L = iCell_CGdata.getNeighbor_ghost(off_L, CGdata);
      CellIndex iCell_R = iCell_CGdata.getNeighbor_ghost(off_R, CGdata);
      real_t phy_L = get_value( CGdata, iCell_L, CG_PHI, off_L );
      real_t phy_R = get_value( CGdata, iCell_R, CG_PHI, off_R );

      // If neighbor is bigger h (which was dx_small) becomes ( dx_small/2 + dx_big/2 = 3/2*dx_small )
      real_t hl = size[dir];
      if( iCell_L.level_diff()==1 ) hl *= 1.5;
      if( iCell_L.level_diff()==-1 ) hl *= 0.75;
      real_t hr = size[dir];
      if( iCell_R.level_diff()==1 ) hr *= 1.5;
      if( iCell_R.level_diff()==-1 ) hr *= 0.75;
      
      real_t dphy_L = (phy_L - phy_C)/hl;
      real_t dphy_R = (phy_C - phy_R)/hr;

      if( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) dphy_L = 0;
      if( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) dphy_R = 0;

      Uout.at(iCell_Uout, (VarIndex)(IGX+dir)) = (dphy_L + dphy_R)/2; 
    };
    gradient(IX);
    gradient(IY);
    if(ndim==3) gradient(IZ);

  });

  pdata->timers.get("GravitySolver_cg").stop();
}

}// namespace dyablo
}// namespace muscl_block

FACTORY_REGISTER( dyablo::muscl_block::GravitySolverFactory, dyablo::muscl_block::GravitySolver_cg, "GravitySolver_cg" );
