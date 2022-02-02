#include "refine_condition/RefineCondition_base.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"

namespace dyablo {
namespace muscl_block {
namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray = ForeachCell::CellArray_global;
using PatchArray = ForeachCell::CellArray_patch;
using CellIndex = ForeachCell::CellIndex;

}// namespace
}// namespace dyablo
}// namespace muscl_block

#include "hydro/CopyGhostBlockCellData.h"

namespace dyablo {
namespace muscl_block {

namespace{

template< int ndim >
KOKKOS_INLINE_FUNCTION
void compute_primitives(const PatchArray& Ugroup, const CellIndex& iCell_Ugroup, const PatchArray& Qgroup,
                        real_t gamma0, real_t smallr, real_t smallp)
{
  HydroState3d uLoc = getHydroState<ndim>( Ugroup, iCell_Ugroup );
      
  // get primitive variables in current cell
  HydroState3d qLoc;
  real_t c = 0.0;
  if(ndim==3)
    computePrimitives(uLoc, &c, qLoc, gamma0, smallr, smallp);
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
    computePrimitives(uLoc_2d, &c, qLoc_2d, gamma0, smallr, smallp);
    copy_state(qLoc, qLoc_2d);
  }

  setHydroState<ndim>( Qgroup, iCell_Ugroup, qLoc );
}

constexpr real_t eps = std::numeric_limits<real_t>::epsilon();

KOKKOS_INLINE_FUNCTION
real_t second_derivative_error(
  const PatchArray& Qgroup, const CellIndex& iCell, 
  VarIndex ivar, ComponentIndex3D dir)
{
  constexpr real_t epsref = 0.01;  

  CellIndex::offset_t offset_l{}, offset_r{};
  offset_l[dir] = -1;
  offset_r[dir] = +1;

  real_t ql = Qgroup.at(iCell + offset_l, ivar);
  real_t qc = Qgroup.at(iCell, ivar);
  real_t qr = Qgroup.at(iCell + offset_r, ivar);

  const real_t fr = qr - qc;    
  const real_t fl = ql - qc;
  
  const real_t fc = FABS(qr) + FABS(ql) + 2 * FABS(qc);

  return FABS(fr + fl) / (FABS(fr) + FABS(fl) + epsref * fc + eps);
}

}

class RefineCondition_second_derivative_error : public RefineCondition
{
public:
  RefineCondition_second_derivative_error( ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : foreach_cell(foreach_cell),
      timers(timers),
      error_min ( configMap.getValue<real_t>("amr", "error_min", 0.2) ),
      error_max ( configMap.getValue<real_t>("amr", "error_max", 0.8) ),
      gravity_type( configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE) ),
      xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ),
      ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ),
      zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ),
      xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
      ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
      zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
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

  void mark_cells( const ForeachCell::CellArray_global_ghosted& Uin )
  {
    int ndim = foreach_cell.getDim();
    if( ndim == 2 )
      mark_cells_aux<2>( Uin );
    else if( ndim == 3 )
      mark_cells_aux<3>( Uin );
  }

  template< int ndim >
  void mark_cells_aux( const ForeachCell::CellArray_global_ghosted& Uin )
  {
    // TODO : only keep VarIndex relevant for markers computation
    FieldManager fm_refvar({ID, IP, IU, IV, IW});
    auto fm = fm_refvar.get_id2index();
    int nbfields = fm_refvar.nbfields();

    BoundaryConditionType xbound = bxmin; assert(bxmin == bxmax);
    BoundaryConditionType ybound = bymin; assert(bymin == bymax);
    BoundaryConditionType zbound = bzmin; assert(bzmin == bzmax);
    real_t xmin = this->xmin;
    real_t ymin = this->ymin;
    real_t zmin = this->zmin;
    real_t xmax = this->xmax;
    real_t ymax = this->ymax;
    real_t zmax = this->zmax;
    real_t gamma0 = this->gamma0;
    real_t smallr = this->smallr;
    real_t smallp = this->smallp;
    real_t error_min = this->error_min;
    real_t error_max = this->error_max;

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Create abstract temporary ghosted arrays for patches 
    ForeachCell::CellArray_patch::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, fm, nbfields);
    ForeachCell::CellArray_patch::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, fm, nbfields);

    uint32_t nbOcts = foreach_cell.get_amr_mesh().getNumOctants();
    Kokkos::View<real_t*> oct_err_max("Oct_err_max", nbOcts);

    // Iterate over patches
    foreach_cell.foreach_patch( "RefineCondition_generic::mark_cells",
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      ForeachCell::CellArray_patch Ugroup = patch.allocate_tmp(Ugroup_);
      ForeachCell::CellArray_patch Qgroup = patch.allocate_tmp(Qgroup_);  

      // Copy non ghosted array Uin into temporary ghosted Ugroup with two ghosts
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const ForeachCell::CellIndex& iCell_Ugroup)
      {
          copyGhostBlockCellData<ndim>(
          Uin, iCell_Ugroup, 
          cellmetadata, 
          xmin, ymin, zmin, 
          xmax, ymax, zmax, 
          xbound, ybound, zbound,
          Ugroup);
      });

      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      { 
        compute_primitives<ndim>(Ugroup, iCell_Ugroup, Qgroup, gamma0, smallr, smallp);
      });

      patch.foreach_cell(Uin, CELL_LAMBDA(const CellIndex& iCell_U)
      { 
        CellIndex iCell_Qgroup = Qgroup.convert_index(iCell_U);

        real_t f_max = 0;

        for(VarIndex ivar : {ID,IP})
        {
          real_t fx = second_derivative_error(Qgroup, iCell_Qgroup, ivar, IX);
          real_t fy = second_derivative_error(Qgroup, iCell_Qgroup, ivar, IY);
          real_t fz = (ndim==2) ? 0 : second_derivative_error(Qgroup, iCell_Qgroup, ivar, IZ);

          f_max = FMAX( f_max, FMAX( fx, FMAX(fy, fz) ) );
        }

        Kokkos::atomic_fetch_max( &oct_err_max( iCell_Qgroup.getOct() ), f_max );
      });
    });

    AMRmesh& pmesh = foreach_cell.get_amr_mesh();
    int level_min = pmesh.get_level_min();
    int level_max = pmesh.get_level_max();
    const LightOctree& lmesh = pmesh.getLightOctree();    
    Kokkos::View<uint32_t*> markers_iOct("markers_iOct", nbOcts);
    Kokkos::View<int*> markers_marker("markers_marker", nbOcts);
    uint32_t nb_markers = 0;
    Kokkos::parallel_scan( "MarkOctantsHydroFunctor::compress_markers", nbOcts,
      KOKKOS_LAMBDA( uint32_t iOct, uint32_t& nb_markers, bool final )
    {
      uint8_t level = lmesh.getLevel({iOct,false});
      real_t error = oct_err_max(iOct);

      // -1 means coarsen
      //  0 means don't modify
      // +1 means refine
      int criterion = -1;
      if (error > error_min)
        criterion = criterion < 0 ? 0 : criterion;
      if (error > error_max)
        criterion = criterion < 1 ? 1 : criterion;

      // Don't coarsen/refine out of [level_min, level_max]
      if( level >= level_max && criterion==1 )
        criterion = 0;
      if( level <= level_min && criterion==-1 )
        criterion = 0;

      if( criterion != 0 )
      {
        if( final )
        {
          markers_iOct(nb_markers) = iOct;
          markers_marker(nb_markers) = criterion;
        }
        nb_markers ++;
      }
    }, nb_markers);

    auto markers_iOct_host = Kokkos::create_mirror_view(markers_iOct);
    auto markers_marker_host = Kokkos::create_mirror_view(markers_marker);
    Kokkos::deep_copy( markers_iOct_host, markers_iOct );
    Kokkos::deep_copy( markers_marker_host, markers_marker );

    Kokkos::parallel_for( "MarkOctantsHydroFunctor::set_markers_pablo", 
                        Kokkos::RangePolicy<Kokkos::OpenMP>(0,nb_markers),
                        [&](uint32_t i)
    {
      uint32_t iOct = markers_iOct_host(i);
      int marker = markers_marker_host(i);

      pmesh.setMarker(iOct, marker);
    });
  }

private:
  ForeachCell& foreach_cell;
  Timers& timers;
  real_t error_min, error_max;

  GravityType gravity_type;
  real_t xmin, ymin, zmin;
  real_t xmax, ymax, zmax;
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

FACTORY_REGISTER( dyablo::muscl_block::RefineConditionFactory, dyablo::muscl_block::RefineCondition_second_derivative_error, "RefineCondition_second_derivative_error" );