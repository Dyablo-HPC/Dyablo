#include "refine_condition/RefineCondition_helper.h"

#include "kokkos_shared.h"
#include "foreach_cell/ForeachCell.h"
#include "utils_hydro.h"
#include "boundary_conditions/BoundaryConditions.h"

namespace dyablo {

namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray = ForeachCell::CellArray_global;
using PatchArray = ForeachCell::CellArray_patch;
using CellIndex = ForeachCell::CellIndex;

}// namespace
}// namespace dyablo


#include "hydro/CopyGhostBlockCellData.h"

namespace dyablo {


namespace{

template< 
  int ndim,
  typename PrimState,
  typename ConsState >
KOKKOS_INLINE_FUNCTION
void compute_primitives(const PatchArray& Ugroup, const CellIndex& iCell_Ugroup, const PatchArray& Qgroup,
                        real_t gamma0, real_t smallr, real_t smallp)
{
  ConsState uLoc{};
  getConservativeState<ndim>( Ugroup, iCell_Ugroup, uLoc );
      
  // get primitive variables in current cell
  PrimState qLoc{};
  real_t c = 0.0;
  computePrimitives<PrimState, ConsState>(uLoc, &c, qLoc, gamma0, smallr, smallp);
  setPrimitiveState<ndim>( Qgroup, iCell_Ugroup, qLoc );
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
      //timers(timers),
      error_min ( configMap.getValue<real_t>("amr", "error_min", 0.2) ),
      error_max ( configMap.getValue<real_t>("amr", "error_max", 0.8) ),
      gravity_type( configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE) ),
      bc_manager( configMap ),
      gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
      smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
      smallc( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
      smallp( smallc*smallc/gamma0 )
  {}

  void mark_cells( const UserData& Uin )
  {
    int ndim = foreach_cell.getDim();
    if( ndim == 2 )
      mark_cells_aux<2, PrimHydroState, ConsHydroState>( Uin );
    else if( ndim == 3 )
      mark_cells_aux<3, PrimHydroState, ConsHydroState>( Uin );
  }

  template< int ndim,
            typename PrimState,
            typename ConsState >
  void mark_cells_aux( const UserData& Uin_ )
  {
    auto bc_manager = this->bc_manager;
    real_t gamma0 = this->gamma0;
    real_t smallr = this->smallr;
    real_t smallp = this->smallp;
    real_t error_min = this->error_min;
    real_t error_max = this->error_max;

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    // Create abstract temporary ghosted arrays for patches 
    ForeachCell::CellArray_patch::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, ConsState::getFieldManager().get_id2index(), ConsState::getFieldManager().nbfields());
    ForeachCell::CellArray_patch::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, PrimState::getFieldManager().get_id2index(), ConsState::getFieldManager().nbfields());

    uint32_t nbOcts = foreach_cell.get_amr_mesh().getNumOctants();
    Kokkos::View<int*> oct_marker_max("Oct_marker_max", nbOcts);

    const UserData::FieldAccessor Uin = Uin_.getAccessor( ConsState::getFieldsInfo() );

    // Iterate over patches
    foreach_cell.foreach_patch( "RefineCondition_generic::mark_cells",
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      ForeachCell::CellArray_patch Ugroup = patch.allocate_tmp(Ugroup_);
      ForeachCell::CellArray_patch Qgroup = patch.allocate_tmp(Qgroup_);  

      // Copy non ghosted array Uin into temporary ghosted Ugroup with two ghosts
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const ForeachCell::CellIndex& iCell_Ugroup)
      {
          copyGhostBlockCellData<ndim, HydroState>(
          Uin, iCell_Ugroup, 
          cellmetadata, 
          bc_manager,
          Ugroup);
      });

      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      { 
        compute_primitives<ndim, PrimState, ConsState>(Ugroup, iCell_Ugroup, Qgroup, gamma0, smallr, smallp);
      });

      patch.foreach_cell(Uin.getShape(), CELL_LAMBDA(const CellIndex& iCell_U)
      { 
        CellIndex iCell_Qgroup = Qgroup.convert_index(iCell_U);

        real_t f_max = 0;

        for(VarIndex ivar : {PrimState::VarIndex::Irho,PrimState::VarIndex::Ip})
        {
          real_t fx = second_derivative_error(Qgroup, iCell_Qgroup, ivar, IX);
          real_t fy = second_derivative_error(Qgroup, iCell_Qgroup, ivar, IY);
          real_t fz = (ndim==2) ? 0 : second_derivative_error(Qgroup, iCell_Qgroup, ivar, IZ);

          f_max = FMAX( f_max, FMAX( fx, FMAX(fy, fz) ) );
        }

        int criterion;
        if( f_max > error_max )
          criterion = RefineCondition::REFINE;
        else if( f_max <= error_min )
          criterion = RefineCondition::COARSEN;
        else
          criterion = RefineCondition::NOCHANGE;

        Kokkos::atomic_fetch_max( &oct_marker_max( iCell_Qgroup.getOct() ), criterion );
      });
    });

    RefineCondition_utils::set_markers(foreach_cell.get_amr_mesh(), oct_marker_max);
  }

private:
  ForeachCell& foreach_cell;
  real_t error_min, error_max;

  GravityType gravity_type;
  BoundaryConditions bc_manager;
  real_t gamma0, smallr, smallc, smallp;
  
};


} // namespace dyablo 

FACTORY_REGISTER( dyablo::RefineConditionFactory, dyablo::RefineCondition_second_derivative_error, "RefineCondition_second_derivative_error" );