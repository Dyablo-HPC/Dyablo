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

    // Computing mu at local position
    auto pos = cellmetadata.getCellCenter(iCell_Qgroup);
    const real_t mu_cell = compute_mu<ndim>(pos);

    PrimState::VarIndex IU = PrimState::VarIndex::Iu;
    PrimState::VarIndex IV = PrimState::VarIndex::Iv;
    PrimState::VarIndex IW = PrimState::VarIndex::Iw;

    auto stencil = [&](const offset_t& offset, PrimState::VarIndex& ivar)
    {
      return Qgroup.at( iCell_Qgroup + offset, ivar );
    };

    ConsHydroState vf{};

    // For each side and each direction
    for( int side : {0,1} )
    {
      int8_t R = side;
      int8_t L = side-1;

      // viscous_flux for direction IX
      {
        const real_t dudx = inv_size[IX] * (stencil({R, 0, 0}, IU) - stencil({L, 0, 0}, IU));
        const real_t dvdx = inv_size[IX] * (stencil({R, 0, 0}, IV) - stencil({L, 0, 0}, IV));

        const real_t dudy = 0.25 * inv_size[IY] * ( stencil({R, 1, 0}, IU) - stencil({R, -1, 0}, IU) 
                                                  + stencil({L, 1, 0}, IU) - stencil({L, -1, 0}, IU));
        const real_t dvdy = 0.25 * inv_size[IY] * ( stencil({R, 1, 0}, IV) - stencil({R, -1, 0}, IV) 
                                                  + stencil({L, 1, 0}, IV) - stencil({L, -1, 0}, IV));

        real_t dwdx=0.0, dudz=0.0, dwdz=0.0, qi_w=0;
        if (ndim == 3) 
        {
          dwdx = inv_size[IX] * ( stencil({R, 0, 0}, IW) - stencil({L, 0, 0}, IW) );
          dudz = 0.25 * inv_size[IZ] * (  stencil({R, 0, 1}, IU) - stencil({R, 0, -1}, IU) 
                                        + stencil({L, 0, 1}, IU) - stencil({L, 0, -1}, IU) );
        
          dwdz = 0.25 * inv_size[IZ] * (  stencil({R, 0, 1}, IW) - stencil({R, 0, -1}, IW) 
                                        + stencil({L, 0, 1}, IW) - stencil({L, 0, -1}, IW) );

          qi_w = 0.5 * ( stencil({R, 0, 0}, IW) + stencil({L, 0, 0}, IW) );
        }

        const real_t qi_u = 0.5 * ( stencil({R, 0, 0}, IU) + stencil({L, 0, 0}, IU) );
        const real_t qi_v = 0.5 * ( stencil({R, 0, 0}, IV) + stencil({L, 0, 0}, IV) );

        // Building relevant tensor elements
        const real_t tau_xx = 4.0/3.0 * dudx - 2.0/3.0 * (dvdy + dwdz);
        const real_t tau_xy = dudy + dvdx;
        const real_t tau_xz = dudz + dwdx;

        // Building flux
        real_t sign = (side == 0 ? -1.0 : 1.0);
        real_t area = size[IY] * size[IZ];
        vf.rho_u += sign * area * tau_xx;
        vf.rho_v += sign * area * tau_xy;
        vf.rho_w += sign * area * tau_xz;
        vf.e_tot += sign * area * (qi_u * tau_xx + qi_v * tau_xy + qi_w * tau_xz);
      }

      // viscous_flux for direction IY
      {
        const real_t dudy = inv_size[IY] * (stencil({0, R, 0}, IU) - stencil({0, L, 0}, IU));
        const real_t dvdy = inv_size[IY] * (stencil({0, R, 0}, IV) - stencil({0, L, 0}, IV));

        const real_t dudx = 0.25 * inv_size[IX] * ( stencil({1, R, 0}, IU) - stencil({-1, R, 0}, IU) 
                                                  + stencil({1, L, 0}, IU) - stencil({-1, L, 0}, IU));
        const real_t dvdx = 0.25 * inv_size[IX] * ( stencil({1, R, 0}, IV) - stencil({-1, R, 0}, IV) 
                                                  + stencil({1, L, 0}, IV) - stencil({-1, L, 0}, IV));

        real_t dwdy=0.0, dvdz=0.0, dwdz=0.0, qi_w=0;
        if (ndim == 3) 
        {
          dwdy = inv_size[IY] * ( stencil({0, R, 0}, IW) - stencil({0, L, 0}, IW) );
          dvdz = 0.25 * inv_size[IZ] * (  stencil({0, R, 1}, IV) - stencil({0, R, -1}, IV) 
                                        + stencil({0, L, 1}, IV) - stencil({0, L, -1}, IV) );
        
          dwdz = 0.25 * inv_size[IZ] * (  stencil({0, R, 1}, IW) - stencil({0, R, -1}, IW) 
                                        + stencil({0, L, 1}, IW) - stencil({0, L, -1}, IW) );

          qi_w = 0.5 * ( stencil({0, R, 0}, IW) + stencil({0, L, 0}, IW) );
        }

        const real_t qi_u = 0.5 * ( stencil({0, R, 0}, IU) + stencil({0, L, 0}, IU) );
        const real_t qi_v = 0.5 * ( stencil({0, R, 0}, IV) + stencil({0, L, 0}, IV) );

        // Building relevant tensor elements
        const real_t tau_yy = 4.0/3.0 * dvdy - 2.0/3.0 * (dudx + dwdz);
        const real_t tau_xy = dudy + dvdx;
        const real_t tau_yz = dvdz + dwdy;

        // Building flux
        real_t sign = (side == 0 ? -1.0 : 1.0);
        real_t area = size[IX] * size[IZ];
        vf.rho_u += sign * area * tau_xy;
        vf.rho_v += sign * area * tau_yy;
        vf.rho_w += sign * area * tau_yz;
        vf.e_tot += sign * area * (qi_u * tau_xy + qi_v * tau_yy + qi_w * tau_yz);
      }

      // viscous_flux for direction IZ
      if (ndim == 3) 
      {
        const real_t dudz = inv_size[IZ] * (stencil({0, 0, R}, IU) - stencil({0, 0, L}, IU));
        const real_t dvdz = inv_size[IZ] * (stencil({0, 0, R}, IV) - stencil({0, 0, L}, IV));
        const real_t dwdz = inv_size[IZ] * (stencil({0, 0, R}, IW) - stencil({0, 0, L}, IW));

        const real_t dudx = 0.25 * inv_size[IX] * ( stencil({1, 0, R}, IU) - stencil({-1, 0, R}, IU) 
                                                  + stencil({1, 0, L}, IU) - stencil({-1, 0, L}, IU));
        const real_t dwdx = 0.25 * inv_size[IX] * ( stencil({1, 0, R}, IW) - stencil({-1, 0, R}, IW) 
                                                  + stencil({1, 0, L}, IW) - stencil({-1, 0, L}, IW));

        const real_t dvdy = 0.25 * inv_size[IY] * ( stencil({0, 1, R}, IV) - stencil({ 0,-1, R}, IV) 
                                                  + stencil({0, 1, L}, IV) - stencil({ 0,-1, L}, IV));
        const real_t dwdy = 0.25 * inv_size[IY] * ( stencil({0, 1, R}, IW) - stencil({ 0,-1, R}, IW) 
                                                  + stencil({0, 1, L}, IW) - stencil({ 0,-1, L}, IW));

        const real_t qi_u = 0.5 * ( stencil({0, 0, R}, IU) + stencil({0, 0, L}, IU) );
        const real_t qi_v = 0.5 * ( stencil({0, 0, R}, IV) + stencil({0, 0, L}, IV) );
        const real_t qi_w = 0.5 * ( stencil({0, 0, R}, IW) + stencil({0, 0, L}, IW) );

        // Building relevant tensor elements
        const real_t tau_zz = 4.0/3.0 * dwdz - 2.0/3.0 * (dudx + dvdy);
        const real_t tau_xz = dudz + dwdx;
        const real_t tau_yz = dvdz + dwdy;

        // Building flux
        real_t sign = (side == 0 ? -1.0 : 1.0);
	real_t area = size[IX] * size[IY];
        vf.rho_u += sign * area * tau_xz;
        vf.rho_v += sign * area * tau_yz;
        vf.rho_w += sign * area * tau_zz;
        vf.e_tot += sign * area * (qi_u * tau_xz + qi_v * tau_yz + qi_w * tau_zz);
      }
    }

    const real_t volume  = size[IX] * size[IY] * (ndim == 3 ? size[IZ] : 1.0);
    const ConsHydroState res = mu_cell * vf / volume;

    setConservativeState<ndim>(rhs, iCell_rhs, res);
  }

private:
  RiemannParams params;
  real_t mu_cst;

};

}
