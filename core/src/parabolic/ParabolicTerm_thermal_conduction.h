#pragma once

#include "UserData.h"
#include "FieldManager.h"
#include "RiemannSolvers.h"
#include "hydro/HydroUpdate_utils.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

#include "boundary_conditions/BoundaryConditions.h"
#include "utils/config/named_enum.h"

/**
 * In the case we have an analytical kappa, this enum allows to pick which 
 * way kappa is calculated.
 */
enum KappaMode {
  KM_NONE,
  KM_TRI_LAYER, 
};

template<>
inline named_enum<KappaMode>::init_list named_enum<KappaMode>::names() 
{
  return {
    {KappaMode::KM_NONE,      "none"},
    {KappaMode::KM_TRI_LAYER, "tri_layer"},
  };
}

namespace dyablo {

namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray  = ForeachCell::CellArray_global;
using PatchArray   = ForeachCell::CellArray_patch;
using CellIndex    = ForeachCell::CellIndex;
using CellMetaData = ForeachCell::CellMetaData;
using pos_t        = Kokkos::Array<real_t, 3>;

enum VarIndex_thermal {
  IRHO, 
  IPRS
};
}

/**
 * @brief Parabolic term solving for thermal conduction : dE/dt + div(Kappa grad T) = 0
 */
class ParabolicTerm_thermal_conduction {
public:
  using State     = HydroState; 
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  ParabolicTerm_thermal_conduction(ConfigMap &configMap)
    : params(configMap),
      bc_manager(configMap),
      kappa_cst(configMap.getValue<real_t>("thermal_conduction", "kappa", 0.0)),
      diffusivity_mode(configMap.getValue<DiffusivityMode>("thermal_conduction", "diffusivity_mode", DM_CONSTANT))
      {
        if (diffusivity_mode == DM_ANALYTICAL) {
          kappa_mode = configMap.getValue<KappaMode>("thermal_conduction", "kappa_mode", KM_NONE);

          if (kappa_mode == KM_TRI_LAYER) {
            tr_thick = configMap.getValue<real_t>("tri_layer", "transition_thickness", 0.0);
            z1       = configMap.getValue<real_t>("tri_layer", "z1", 0.0);
            z2       = configMap.getValue<real_t>("tri_layer", "z2", 1.0);
            K1       = configMap.getValue<real_t>("tri_layer", "K1", 1.0);
            K2       = configMap.getValue<real_t>("tri_layer", "K2", 1.0);
          }
        }
        else 
          kappa_mode = KM_NONE;
      };


  template <int ndim>
  KOKKOS_INLINE_FUNCTION
  real_t compute_kappa(const pos_t& pos) const {
    real_t kappa = kappa_cst;

    // Analytical calculations 
    if (diffusivity_mode == DM_ANALYTICAL) {
      if (kappa_mode == KM_TRI_LAYER) {
        const real_t d = (ndim == 2 ? pos[IY] : pos[IZ]);
        const real_t th = tr_thick;
        const real_t k1 = kappa_cst * K1;
        const real_t k2 = kappa_cst * K2;

        const real_t tr1 = (tanh((d-z1)/th) + 1.0) * 0.5;
        const real_t tr2 = (tanh((z2-d)/th) + 1.0) * 0.5;
        const real_t tr = tr1*tr2;

        kappa = k2 * (1.0-tr) + k1 * tr;
      }
    }

    return kappa;
  }

  static std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo()
  {
    return ConsHydroState::getFieldsInfo();
  }


  template <int ndim, typename Uin_t>
  KOKKOS_INLINE_FUNCTION
  void compute_rhs(const Uin_t&        Uin,
                   const PatchArray&   Ugroup,
                   const PatchArray&   Qgroup,
                   const PatchArray&   rhs,
                   const CellIndex&    iCell_Uout,
                   const CellIndex&    iCell_rhs,
                   const CellMetaData& cellmetadata) const
  {
    // Aliases
    using offset_t = CellIndex::offset_t;
    
    // TODO : Move this to a more appropriate place with EOS
    auto compute_temperature = [](const PrimHydroState &q) -> real_t {
      return q.p / q.rho;
    };

    // Getting cell info
    CellIndex iCell_Uin    = Uin.getShape().convert_index(iCell_Uout);
    CellIndex iCell_Ugroup = Ugroup.convert_index(iCell_rhs);
    auto size = cellmetadata.getCellSize(iCell_Uin);
    auto pos  = cellmetadata.getCellCenter(iCell_Uin);
    real_t V  = size[IX] * size[IY] * (ndim == 3 ? size[IZ] : 1.0);

    PrimHydroState qC;
    getPrimitiveState<ndim>(Qgroup, iCell_Ugroup, qC);
    auto TC = compute_temperature(qC);
    auto kappaC = compute_kappa<ndim>(pos);

    // Relative sizes as function of level difference
    constexpr real_t S[3] {0.75, 1.0, 1.5};

    // Interface area array
    const real_t A[3] = {ndim == 3 ? size[IY]*size[IZ] : size[IY],
                         ndim == 3 ? size[IX]*size[IZ] : size[IX],
                         size[IX]*size[IY]};

    auto compute_thermal_flux = [&](ComponentIndex3D dir) -> real_t {
      offset_t offsetm{0}, offsetp{0};
      offsetm[dir] = -1;
      offsetp[dir] =  1;
      real_t area = A[dir];

      // Getting neighbor element
      auto iiL = iCell_Uin.getNeighbor_ghost(offsetm, Uin);
      auto iiR = iCell_Uin.getNeighbor_ghost(offsetp, Uin);

      // And level differences
      int ldiff_L = iiL.level_diff();
      int ldiff_R = iiR.level_diff();

      // And relative sizes
      real_t SL = S[ldiff_L+1];
      real_t SR = S[ldiff_R+1];

      real_t flux_out = 0.0;
      real_t FL = 0.0;
      real_t FR = 0.0;

      real_t TL, TR;

      // LEFT :
      // Only one neighbor
      if (ldiff_L >= 0) {
        auto pos = cellmetadata.getCellCenter(iCell_Ugroup + offsetm);
        auto kappa = 0.5 * (kappaC + compute_kappa<ndim>(pos));

        PrimHydroState qL; 
        getPrimitiveState<ndim>(Qgroup, iCell_Ugroup + offsetm, qL);
        TL = compute_temperature(qL);

        FL = kappa * (TC - TL) / (SL * size[dir]);
      }
      // Multiple neighbors
      else {
        constexpr real_t nfac = (ndim == 2 ? 0.5 : 0.25);
        real_t tmp_flux{0.0};
        
        foreach_smaller_neighbor<ndim>(iiL, offsetm, Uin.getShape(), 
          [&](const CellIndex& iCell_neighbor)
            {
              ConsHydroState uL;
              getConservativeState<ndim>(Uin, iCell_neighbor, uL);
              PrimHydroState qL = consToPrim<ndim>(uL, params.gamma0);
              const real_t TL = compute_temperature(qL);

              auto pos = cellmetadata.getCellCenter(iCell_neighbor);
              auto kappa = 0.5 * (kappaC + compute_kappa<ndim>(pos));

              tmp_flux += kappa * (TC - TL);
            });
        FL += nfac * tmp_flux / (SL * size[dir]);
      }

      // RIGHT :
      // Only one neighbor
      if (ldiff_R >= 0) {
        auto pos = cellmetadata.getCellCenter(iCell_Ugroup + offsetp);
        auto kappa = 0.5 * (kappaC + compute_kappa<ndim>(pos));

        PrimHydroState qR;
        getPrimitiveState<ndim>(Qgroup, iCell_Ugroup + offsetp, qR);
        TR = compute_temperature(qR);

        FR = kappa * (TR - TC) / (SR * size[dir]);
      }
      // Multiple neighbors
      else {
        constexpr real_t nfac = (ndim == 2 ? 0.5 : 0.25);
        real_t tmp_flux{0.0};
        
        foreach_smaller_neighbor<ndim>(iiR, offsetp, Uin.getShape(), 
          [&](const CellIndex& iCell_neighbor)
            {
              ConsHydroState uR;
              getConservativeState<ndim>(Uin, iCell_neighbor, uR);
              PrimHydroState qR = consToPrim<ndim>(uR, params.gamma0);
              const real_t TR = compute_temperature(qR);

              auto pos = cellmetadata.getCellCenter(iCell_neighbor);
              auto kappa = 0.5 * (kappaC + compute_kappa<ndim>(pos));

              tmp_flux += kappa * (TR - TC);
            });
        FR += nfac * tmp_flux / (SR * size[dir]);
      }

      if (iiL.is_boundary() && bc_manager.bc_min[dir] == BC_USER)
        FL = bc_manager.overrideBoundaryHeatFlux<ndim, State>(FL, qC, kappaC, size[dir], dir, true);
      if (iiR.is_boundary() && bc_manager.bc_max[dir] == BC_USER)
        FR = bc_manager.overrideBoundaryHeatFlux<ndim, State>(FR, qC, kappaC, size[dir], dir, false);

      flux_out = area * (FR - FL);

      return flux_out;
    };

    real_t tf_x = compute_thermal_flux(IX);
    real_t tf_y = compute_thermal_flux(IY);
    real_t tf_z = (ndim == 3 ? compute_thermal_flux(IZ) : 0.0);

    // Storing results
    ConsHydroState res{};
    res.e_tot = (tf_x + tf_y + tf_z) / V;

    setConservativeState<ndim>(rhs, iCell_rhs, res);
  }

private:
  RiemannParams params;
  BoundaryConditions bc_manager;

  real_t kappa_cst;
  DiffusivityMode diffusivity_mode;
  KappaMode kappa_mode;

  // Tri-Layer parameters
  real_t tr_thick, z1, z2, K1, K2;
};

}