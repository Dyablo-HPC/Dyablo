#include "GravitySolver_cg.h"

#include "utils/monitoring/Timers.h"
#include "mpi/GhostCommunicator.h"
#include "foreach_cell/ForeachCell_utils.h"
#include <mpi.h>

namespace dyablo { 

struct GravitySolver_cg::Data{
  ForeachCell& foreach_cell;
  
  Timers& timers;  

  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;

  Kokkos::Array<BoundaryConditionType, 3> boundarycondition;

  real_t gravity_constant = 1;
  real_t CG_eps;
  bool print_cg_iter;
};

GravitySolver_cg::GravitySolver_cg(
  ConfigMap& configMap,
  ForeachCell& foreach_cell,
  Timers& timers )
 : pdata(new Data
    {
      foreach_cell,
      timers,
      configMap.getValue<real_t>("mesh", "xmin", 0.0),
      configMap.getValue<real_t>("mesh", "ymin", 0.0),
      configMap.getValue<real_t>("mesh", "zmin", 0.0),
      configMap.getValue<real_t>("mesh", "xmax", 1.0),
      configMap.getValue<real_t>("mesh", "ymax", 1.0),
      configMap.getValue<real_t>("mesh", "zmax", 1.0),
      {
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
        configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)
      },
      configMap.getValue<real_t>("gravity", "4_Pi_G", 1.0), // 4*Pi*G
      configMap.getValue<real_t>("gravity", "CG_eps", 1E-3),  // target ||r||/||b|| for conjugate gradient
      configMap.getValue<bool>("gravity", "print_cg_iter", false) 
    })
{
  int ndim = configMap.getValue<int>("mesh", "ndim", 3);
  if( ndim != 3 )
    throw std::runtime_error("GravitySolver_cg can only run in 3D");
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
//constexpr VarIndex CG_IZ = (VarIndex)3;

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
    int nbCells =
    foreach_smaller_neighbor<ndim, false>( 
      iCell_U, offset, U, 
      [&](const ForeachCell::CellIndex& iCell_ghost)
    {
      sum += U.at(iCell_ghost, var);
    });
    return sum/nbCells;
  } 
}

KOKKOS_INLINE_FUNCTION
real_t matprod(const GhostedArray& GCdata, const CellIndex& iCell_CGdata, VarIndex var, real_t dx, real_t dy, real_t dz, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition)
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

KOKKOS_INLINE_FUNCTION
real_t Aii(const GhostedArray& GCdata, const CellIndex& iCell_CGdata, real_t dx, real_t dy, real_t dz, const Kokkos::Array<BoundaryConditionType, 3>& boundarycondition)
{
  return (dx*dy*dz)*(2/(dx*dx)+2/(dy*dy)+2/(dz*dz));
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
  const ForeachCell::CellArray_global_ghosted& Uin,
  const ForeachCell::CellArray_global_ghosted& Uout )
{
  uint8_t ndim = pdata->foreach_cell.getDim();
  ForeachCell& foreach_cell = pdata->foreach_cell;
  real_t xmin = pdata->xmin, ymin = pdata->ymin, zmin = pdata->zmin;
  real_t xmax = pdata->xmax, ymax = pdata->ymax, zmax = pdata->zmax;
  Kokkos::Array<BoundaryConditionType, 3> boundarycondition = pdata->boundarycondition;
  GhostCommunicator ghost_comm(std::shared_ptr<AMRmesh>(&foreach_cell.get_amr_mesh(), [](AMRmesh*){}));

  real_t gravity_constant = pdata->gravity_constant;
  real_t eps = pdata->CG_eps;

  ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

  // Setup field manager for CGdata
  FieldManager fieldManager( {CG_IP, CG_IR, CG_PHI/*, CG_IZ*/} );
  // Allocate global array
  GhostedArray CGdata = foreach_cell.allocate_ghosted_array( "CGdata", fieldManager );

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

  // Compute initial residual
  real_t R=0, B=0, r_dot_z=0;
  foreach_cell.reduce_cell( "Gravity_cg::init_gc", CGdata, 
    CELL_LAMBDA(const CellIndex& iCell_CGdata, real_t& update_r, real_t& update_b, real_t& update_r_dot_z )
  {
    // Get gravity potential from last iteration as approximate solution
    CGdata.at(iCell_CGdata, CG_PHI) = Uin.at(iCell_CGdata, IGPHI);
    auto cell_size = cells.getCellSize(iCell_CGdata);
    real_t bi = b(Uin, iCell_CGdata, rho_mean, gravity_constant, cell_size);
    real_t r = bi - matprod(Uin, iCell_CGdata, IGPHI, cell_size[IX], cell_size[IY], cell_size[IZ], boundarycondition);
    CGdata.at(iCell_CGdata, CG_IR) = r; // r = b-A*x0
    real_t z = r/Aii(Uin, iCell_CGdata, cell_size[IX], cell_size[IY], cell_size[IZ], boundarycondition); // z = M^(-1)r
    //CGdata.at(iCell_CGdata, CG_IZ) = z;
    CGdata.at(iCell_CGdata, CG_IP) = z; // p <- r
    update_r += r*r ;
    update_b += bi*bi ;
    update_r_dot_z += r*z;
  }, R, B, r_dot_z );  
  R = MPI_Allreduce_scalar(R);
  B = MPI_Allreduce_scalar(B);
  r_dot_z = MPI_Allreduce_scalar(r_dot_z);
  
  if(pdata->print_cg_iter)
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
    // alpha <- rz/p'Ap
    // phi   <- phi + alpha * p
    // r     <- r   - alpha * Ap
    // Rnext <- ||r||
    real_t Rnext = 0, r_dot_z_next = 0;
    foreach_cell.reduce_cell( "Gravity_cg::alpha", CGdata, 
      CELL_LAMBDA(const CellIndex& iCell_CGdata, real_t& update_Rnext, real_t& update_r_dot_z_next)
    {
      auto cell_size = cells.getCellSize(iCell_CGdata);
      real_t alpha = r_dot_z/pAp;
      CGdata.at(iCell_CGdata, CG_PHI) += alpha*CGdata.at(iCell_CGdata, CG_IP);
      real_t r = CGdata.at(iCell_CGdata, CG_IR) - alpha*matprod(CGdata, iCell_CGdata, CG_IP, cell_size[IX], cell_size[IY], cell_size[IZ], boundarycondition);
      CGdata.at(iCell_CGdata, CG_IR) = r;
      real_t z = r/Aii(Uin, iCell_CGdata, cell_size[IX], cell_size[IY], cell_size[IZ], boundarycondition); // z = M^(-1)r
      //CGdata.at(iCell_CGdata, CG_IZ) = z;
      update_Rnext += r*r;
      update_r_dot_z_next += r * z;
    }, Rnext, r_dot_z_next);
    Rnext = MPI_Allreduce_scalar(Rnext);
    r_dot_z_next = MPI_Allreduce_scalar(r_dot_z_next);

    // Update p <- z + r_dot_z_next/r_dot_z * p
    foreach_cell.foreach_cell( "Gravity_cg::beta", CGdata, 
      CELL_LAMBDA(const CellIndex& iCell_CGdata)
    {
      auto cell_size = cells.getCellSize(iCell_CGdata);
      real_t beta = r_dot_z_next/r_dot_z;
      //real_t z = CGdata.at(iCell_CGdata, CG_IZ);
      real_t r = CGdata.at(iCell_CGdata, CG_IR);
      real_t z = r/Aii(Uin, iCell_CGdata, cell_size[IX], cell_size[IY], cell_size[IZ], boundarycondition);
      CGdata.at(iCell_CGdata, CG_IP) = z + beta * CGdata.at(iCell_CGdata, CG_IP);
    });

    R = Rnext;
    r_dot_z = r_dot_z_next;
  }
  if(pdata->print_cg_iter)
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
      real_t phi_C = CGdata.at(iCell_CGdata, CG_PHI);
      Uout.at(iCell_Uout, IGPHI) = phi_C;
      CellIndex::offset_t off_L = {}; off_L[dir] = -1;
      CellIndex::offset_t off_R = {}; off_R[dir] = +1;
      CellIndex iCell_L = iCell_CGdata.getNeighbor_ghost(off_L, CGdata);
      CellIndex iCell_R = iCell_CGdata.getNeighbor_ghost(off_R, CGdata);
      real_t phi_L = get_value( CGdata, iCell_L, CG_PHI, off_L );
      real_t phi_R = get_value( CGdata, iCell_R, CG_PHI, off_R );

      // If neighbor is bigger h (which was dx_small) becomes ( dx_small/2 + dx_big/2 = 3/2*dx_small )
      real_t hl = size[dir];
      if( iCell_L.level_diff()==1 ) hl *= 1.5;
      if( iCell_L.level_diff()==-1 ) hl *= 0.75;
      real_t hr = size[dir];
      if( iCell_R.level_diff()==1 ) hr *= 1.5;
      if( iCell_R.level_diff()==-1 ) hr *= 0.75;
      
      real_t dphi_L = (phi_L - phi_C)/hl;
      real_t dphi_R = (phi_C - phi_R)/hr;

      if( boundarycondition[dir] == BC_ABSORBING && iCell_L.is_boundary() ) dphi_L = 0;
      if( boundarycondition[dir] == BC_ABSORBING && iCell_R.is_boundary() ) dphi_R = 0;

      Uout.at(iCell_Uout, (VarIndex)(IGX+dir)) = (dphi_L + dphi_R)/2; 
    };
    gradient(IX);
    gradient(IY);
    if(ndim==3) gradient(IZ);

  });

  pdata->timers.get("GravitySolver_cg").stop();
}

}// namespace dyablo

FACTORY_REGISTER( dyablo::GravitySolverFactory, dyablo::GravitySolver_cg, "GravitySolver_cg" );