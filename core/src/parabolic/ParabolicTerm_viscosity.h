#pragma once

#include "FieldManager.h"
#include "RiemannSolvers.h"
#include "hydro/HydroUpdate_utils.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo {

namespace{

using GhostedArray = ForeachCell::CellArray_global_ghosted;
using GlobalArray  = ForeachCell::CellArray_global;
using PatchArray   = ForeachCell::CellArray_patch;
using CellIndex    = ForeachCell::CellIndex;
using CellMetaData = ForeachCell::CellMetaData;
}

/**
 * @brief Parabolic term solving for viscosity : dE/dt + div(KappaT) = 0
 */
class ParabolicTerm_viscosity {
public:
  using State     = HydroState;
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  ParabolicTerm_viscosity(ConfigMap &configMap)
    : params(configMap),
      mu_cst(configMap.getValue<real_t>("viscosity", "mu", 0.0)) {};

  template <int ndim>
  KOKKOS_INLINE_FUNCTION
  real_t compute_mu(const pos_t& pos) const {
    real_t mu = 0.0; 
    mu = mu_cst;

    return mu;
  }

  static std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo()
  {
    return ConsHydroState::getFieldsInfo();
  }

  /**
   * @todo : fill in doc
   * @todo : Replace the whole block of code with directions switching by a nice swapComponent at start
   * 
   * @brief 
   * 
   * @tparam ndim 
   * @param Uin 
   * @param Ugroup 
   * @param Qgroup 
   * @param rhs 
   * @param iCell_Uout 
   * @param iCell_rhs 
   * @param cellmetadata 
   * @return KOKKOS_INLINE_FUNCTION 
   */
  template <int ndim, typename U_t>
  KOKKOS_INLINE_FUNCTION
  void compute_rhs(const U_t&          Uin,
                   const PatchArray&   Ugroup,
                   const PatchArray&   Qgroup,
                   const PatchArray&   rhs,
                   const CellIndex&    iCell_Uout,
                   const CellIndex&    iCell_rhs,
                   const CellMetaData& cellmetadata) const
  {
    // Aliases
    using offset_t = CellIndex::offset_t;
    using pos_t    = CellMetaData::pos_t;
    auto size = cellmetadata.getCellSize(iCell_Uout);
    pos_t inv_size {1.0/size[IX], 1.0/size[IY], 1.0/size[IZ]};

    // Index in grouped array
    auto iCell_Qgroup = Qgroup.convert_index(iCell_Uout);
    PrimHydroState q;
    getPrimitiveState<ndim>(Qgroup, iCell_Qgroup, q);

    const ComponentIndex3D tangent_comp[3][2] {{IY, IZ},
                                               {IX, IZ},
                                               {IX, IY}};

    const real_t volume {size[IX] * size[IY] * (ndim == 3 ? size[IZ] : 1.0)};
    const real_t A[3] = {ndim == 3 ? size[IY]*size[IZ] : size[IY],
                         ndim == 3 ? size[IX]*size[IZ] : size[IX],
                         size[IX]*size[IY]};

    

    using Stencil = PrimHydroState[3][3][3];
    Stencil stencil{};

    // TODO : Place this in a utility header/source
    // TODO : Should be optimized to only extract the right cells for the stencil
    // Lambda to extract a cubic stencil around an array
    auto fill_stencil = [&](const CellIndex &iCell) -> void {
      offset_t off;
      int k0 = -1;
      int k1 = (ndim == 2 ? 0 : 2);

      for (int i=-1; i<2; ++i) {
        off[IX] = i;
        for (int j=-1; j<2; ++j) {
          off[IY] = j;
          for (int k=k0; k<k1; ++k) {
            off[IZ] = (ndim == 2 ? k+1 : k);

            getPrimitiveState<ndim>(Qgroup, iCell_Qgroup + off, stencil[i+1][j+1][k+1]);
          }
        }
      }
    };


    auto compute_viscous_flux = [&](const ComponentIndex3D dir) {
      // Here i is the normal component, j the 1st tangential component and k the 2nd
      // Also, u is the normal velocity component, v and w the two tangential
      const auto td1 = tangent_comp[dir][0];
      const auto td2 = tangent_comp[dir][1];

      // Computing mu at local position
      auto pos = cellmetadata.getCellCenter(iCell_Qgroup);
      const real_t mu = compute_mu<ndim>(pos);

      real_t area = A[dir];

      ConsHydroState vf{};

      // Extracting cells from stencil
      fill_stencil(iCell_Qgroup);

      // Going on each side of the cell
      for (int side=1; side < 3; ++side) {
        if (dir == IX) {
          PrimHydroState qi = 0.5 * (stencil[side][1][1] + stencil[side-1][1][1]);
          
          const real_t dudx = inv_size[dir] * (stencil[side][1][1].u - stencil[side-1][1][1].u);
          const real_t dvdx = inv_size[dir] * (stencil[side][1][1].v - stencil[side-1][1][1].v);

          const real_t dudy = 0.25 * inv_size[td1] * (stencil[side  ][2][1].u - stencil[side  ][0][1].u 
                                                    + stencil[side-1][2][1].u - stencil[side-1][0][1].u);
          const real_t dvdy = 0.25 * inv_size[td1] * (stencil[side  ][2][1].v - stencil[side  ][0][1].v 
                                                    + stencil[side-1][2][1].v - stencil[side-1][0][1].v);


          real_t dwdx=0.0, dudz=0.0, dwdz=0.0;
          if (ndim == 3) {
            dwdx = inv_size[dir] * (stencil[side][1][1].w - stencil[side-1][1][1].w);
            dudz = 0.25 * inv_size[td2] * (stencil[side  ][1][2].u - stencil[side  ][1][0].u 
                                         + stencil[side-1][1][2].u - stencil[side-1][1][0].u);
          
            dwdz = 0.25 * inv_size[td2] * (stencil[side  ][1][2].w - stencil[side  ][1][0].w 
                                         + stencil[side-1][1][2].w - stencil[side-1][1][0].w);
          }

          // Building relevant tensor elements
          constexpr real_t four_thirds = 4.0/3.0;
          constexpr real_t two_thirds  = 2.0/3.0;
          const real_t tau_xx = four_thirds * dudx - two_thirds * (dvdy + dwdz);
          const real_t tau_xy = dudy + dvdx;
          const real_t tau_xz = dudz + dwdx;

          // Building flux
          real_t sign = (side == 1 ? -1.0 : 1.0);
          vf.rho_u += sign * mu * tau_xx;
          vf.rho_v += sign * mu * tau_xy;
          vf.rho_w += sign * mu * tau_xz;
          vf.e_tot += sign * mu * (qi.u * tau_xx + qi.v * tau_xy + qi.w * tau_xz);
        }
        else if (dir == IY) {
          PrimHydroState qi = 0.5 * (stencil[1][side][1] + stencil[1][side-1][1]);
          
          const real_t dudy = inv_size[dir] * (stencil[1][side][1].u - stencil[1][side-1][1].u);
          const real_t dvdy = inv_size[dir] * (stencil[1][side][1].v - stencil[1][side-1][1].v);

          const real_t dudx = 0.25 * inv_size[td1] * (stencil[2][side  ][1].u - stencil[0][side  ][1].u 
                                                    + stencil[2][side-1][1].u - stencil[0][side-1][1].u);
          const real_t dvdx = 0.25 * inv_size[td1] * (stencil[2][side  ][1].v - stencil[0][side  ][1].v 
                                                    + stencil[2][side-1][1].v - stencil[0][side-1][1].v);

          real_t dwdy=0.0, dvdz=0.0, dwdz=0.0;
          if (ndim == 3) {                                         
            dwdy = inv_size[dir] * (stencil[1][side][1].w - stencil[1][side-1][1].w);
            dvdz = 0.25 * inv_size[td2] * (stencil[1][side  ][2].v - stencil[1][side  ][0].v 
                                         + stencil[1][side-1][2].v - stencil[1][side-1][0].v);
            dwdz = 0.25 * inv_size[td2] * (stencil[1][side  ][2].w - stencil[1][side  ][0].w 
                                         + stencil[1][side-1][2].w - stencil[1][side-1][0].w);
          }

          // Building relevant tensor elements
          constexpr real_t four_thirds = 4.0/3.0;
          constexpr real_t two_thirds  = 2.0/3.0;
          const real_t tau_yy = four_thirds * dvdy - two_thirds * (dudx + dwdz);
          const real_t tau_xy = dudy + dvdx;
          const real_t tau_yz = dvdz + dwdy;

          // Building flux
          real_t sign = (side == 1 ? -1.0 : 1.0);
          vf.rho_u += sign * mu * tau_xy;
          vf.rho_v += sign * mu * tau_yy;
          vf.rho_w += sign * mu * tau_yz;
          vf.e_tot += sign * mu * (qi.u * tau_xy + qi.v * tau_yy + qi.w * tau_yz);
        }
        else {
          PrimHydroState qi = 0.5 * (stencil[1][1][side] + stencil[1][1][side-1]);
          
          const real_t dudz = inv_size[dir] * (stencil[1][1][side].u - stencil[1][1][side-1].u);
          const real_t dvdz = inv_size[dir] * (stencil[1][1][side].v - stencil[1][1][side-1].v);
          const real_t dwdz = inv_size[dir] * (stencil[1][1][side].w - stencil[1][1][side-1].w);

          const real_t dudx = 0.25 * inv_size[td1] * (stencil[2][1][side  ].u - stencil[0][1][side  ].u 
                                                    + stencil[2][1][side-1].u - stencil[0][1][side-1].u);
          const real_t dwdx = 0.25 * inv_size[td1] * (stencil[2][1][side  ].w - stencil[0][1][side  ].w 
                                                    + stencil[2][1][side-1].w - stencil[0][1][side-1].w);
                                                    
          const real_t dvdy = 0.25 * inv_size[td2] * (stencil[1][2][side  ].v - stencil[1][0][side  ].v 
                                                    + stencil[1][2][side-1].v - stencil[1][0][side-1].v);
          const real_t dwdy = 0.25 * inv_size[td2] * (stencil[1][2][side  ].w - stencil[1][0][side  ].w 
                                                    + stencil[1][2][side-1].w - stencil[1][0][side-1].w);

          // Building relevant tensor elements
          constexpr real_t four_thirds = 4.0/3.0;
          constexpr real_t two_thirds  = 2.0/3.0;
          const real_t tau_zz = four_thirds * dwdz - two_thirds * (dudx + dvdy);
          const real_t tau_xz = dudz + dwdx;
          const real_t tau_yz = dvdz + dwdy;

          // Building flux
          real_t sign = (side == 1 ? -1.0 : 1.0);
          vf.rho_u += sign * mu * tau_xz;
          vf.rho_v += sign * mu * tau_yz;
          vf.rho_w += sign * mu * tau_zz;
          vf.e_tot += sign * mu * (qi.u * tau_xz + qi.v * tau_yz + qi.w * tau_zz);
        }
      }
      
      return area * vf;
    };

    const ConsHydroState vf_x = compute_viscous_flux(IX);
    const ConsHydroState vf_y = compute_viscous_flux(IY);
    ConsHydroState vf_z{};
    if (ndim == 3) 
      vf_z = compute_viscous_flux(IZ);
    const ConsHydroState res = mu_cst * (vf_x + vf_y + vf_z) / volume;

    setConservativeState<ndim>(rhs, iCell_rhs, res);
  }

private:
  RiemannParams params;
  real_t mu_cst;

};

}