#include <memory>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "ScalarSimulationData.h"
#include "ParabolicUpdate_base.h"
#include "ParabolicTerm.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"
#include "utils_hydro.h"
#include "utils/config/ConfigMap.h"

#include "parabolic/ParabolicTerm_thermal_conduction.h"
#include "parabolic/ParabolicTerm_viscosity.h"

namespace dyablo {
namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray = ForeachCell::CellArray_global;
using PatchArray = ForeachCell::CellArray_patch;
using CellIndex = ForeachCell::CellIndex;

}// namespace
}

#include "hydro/CopyGhostBlockCellData.h"

namespace dyablo {
/**
 * @brief Kernel implementing an explicit update for a parabolic term
 * 
 * @param configMap The configuration map of the run
 * @param foreach_cell Traversal used for the calculation of the kernel
 * @param timers handles to time kernel application
 * @param term_type type of parabolic term being solved
 */
class ParabolicUpdate_explicit : public ParabolicUpdate {
public:
  ParabolicUpdate_explicit(
    ConfigMap&        configMap,
    ForeachCell&      foreach_cell,
    Timers&           timers, 
    ParabolicTermType term_type)
   : foreach_cell(foreach_cell),
     configMap(configMap),
     timers(timers),
     term_type(term_type), 
     bc_manager(configMap) { 

     }

  void update( UserData& U, ScalarSimulationData& scalar_data) {
    real_t dt = scalar_data.get<real_t>("dt");
    int ndim = foreach_cell.getDim();
    if (ndim == 2)
      update_parabolic<2>(U, dt);
    else
      update_parabolic<3>(U, dt);
  }
  
  template <int ndim>
  void update_parabolic(UserData &U,
                        real_t dt) 
  {
    if (term_type == PARABOLIC_THERMAL_CONDUCTION)
      update_aux<ndim, ParabolicTerm_thermal_conduction>(U, dt);
    else if (term_type == PARABOLIC_VISCOSITY)
      update_aux<ndim, ParabolicTerm_viscosity>(U, dt);
    else
      throw std::runtime_error("Error, parabolic term not implemented");
  }

  template <
    int ndim, 
    typename ParabolicTerm>
  void update_aux(UserData &U, 
                  real_t dt) 
  {
    using PatchArray = ForeachCell::CellArray_patch;
    using State      = typename ParabolicTerm::State;
    using ConsState  = typename ParabolicTerm::ConsState;
    using PrimState  = typename ParabolicTerm::PrimState;

    auto fm_cons = ConsState::getFieldManager().get_id2index();
    auto fm_prim = PrimState::getFieldManager().get_id2index();

    // 1- Allocating temporary arrays
    // Note : Should that be 2, 2, 2 ? or 1, 1, 1 ? or parabolic-term-type dependent ?
    PatchArray::Ref Ugroup_ = foreach_cell.reserve_patch_tmp("Ugroup", 2, 2, (ndim == 3)?2:0, fm_cons, State::N);
    PatchArray::Ref Qgroup_ = foreach_cell.reserve_patch_tmp("Qgroup", 2, 2, (ndim == 3)?2:0, fm_prim, State::N);
    PatchArray::Ref rhs_    = foreach_cell.reserve_patch_tmp("rhs", 0, 0, 0, fm_cons, State::N);

    ParabolicTerm parabolic_term{configMap};

    std::string kernel_name = "Parabolic[explicit] " + named_enum<ParabolicTermType>::to_string(term_type);
    timers.get(kernel_name).start();

    BoundaryConditions bc_manager = this->bc_manager;
    RiemannParams params{configMap};

    auto fields_info = parabolic_term.getFieldsInfo();
    UserData::FieldAccessor Uin = U.getAccessor( fields_info );
    auto fields_info_next = fields_info;
    for( auto& p : fields_info_next )
      p.name += "_next";
    UserData::FieldAccessor Uout = U.getAccessor( fields_info_next );

    // Iteration over all the patches
    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
    foreach_cell.foreach_patch(kernel_name,
      PATCH_LAMBDA( const ForeachCell::Patch& patch) 
    {
      // Allocating temporary patchs
      PatchArray Ugroup = patch.allocate_tmp(Ugroup_);
      PatchArray Qgroup = patch.allocate_tmp(Qgroup_);
      PatchArray rhs    = patch.allocate_tmp(rhs_);

      // 2- Copy non ghosted array Uin into temporary ghosted Ugroup with two ghosts
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      {
          copyGhostBlockCellData<ndim, State>(
          Uin, iCell_Ugroup, 
          cellmetadata, 
          bc_manager,
          Ugroup);
      });

      // 3- Convert to primitives
      patch.foreach_cell(Ugroup, CELL_LAMBDA(const CellIndex& iCell_Ugroup)
      { 
        compute_primitives<ndim, State>(params, Ugroup, iCell_Ugroup, Qgroup);
      });

      // 4- Calculating rhs term and applying parabolic term
      patch.foreach_cell(Uout.getShape(), CELL_LAMBDA(const CellIndex& iCell_Uout)
      {
        CellIndex iCell_rhs = rhs.convert_index(iCell_Uout);
        parabolic_term.template compute_rhs<ndim>(Uin, Ugroup, Qgroup, rhs, iCell_Uout, iCell_rhs, cellmetadata);

        ConsState u0, v_out{};
        getConservativeState<ndim>(Uout, iCell_Uout, u0);
        getConservativeState<ndim>(rhs,  iCell_rhs, v_out);
        
        setConservativeState<ndim>(Uout, iCell_Uout, u0 + dt * v_out);
      });
    });

    timers.get(kernel_name).stop();
  }

  // Attributes
  ForeachCell &foreach_cell;
  ConfigMap   &configMap;
  Timers      &timers;

  ParabolicTermType term_type;

  BoundaryConditions bc_manager;
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::ParabolicUpdateFactory, 
                 dyablo::ParabolicUpdate_explicit, 
                 "ParabolicUpdate_explicit")
