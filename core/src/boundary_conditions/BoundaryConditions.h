#pragma once

#include "kokkos_shared.h"

#include "RiemannSolvers.h"

namespace dyablo {
template<typename State_> class BoundaryConditions {
public:
  using State     = State_;
  using PrimState = typename State::PrimState;
  using ConsState = typename State::ConsState;

  BoundaryConditions(ConfigMap &configMap) :
      params(configMap),
      bc_min{configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING),
             configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING),
             configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING)},
      bc_max{configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING),
             configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING),
             configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING)},
      xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
      xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),      
      ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
      ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
      zmin(configMap.getValue<real_t>("mesh", "zmin", 0.0)),
      zmax(configMap.getValue<real_t>("mesh", "zmax", 1.0))
  {}

  /**
   * \brief Get field values for a cell outside of the simulation domain
   * 
   * This method is usually called by kernels when trying to access neighbors outside
   * of the simulation domain.
   * Returns the state of a virtual cell outside of the domain.
   * The value returned may be determined by the values inside the domain 
   * (e.g. reflecting/absorbing boundary conditions)
   * 
   * \tparam ndim the number of dimensions
   * 
   * \param Uin input array of conservative variables
   * \param iCell_boundary cell index outside of the domain (iCell_boundary.is_boundary() must be true)
   * \param metadata CellMetaData object allowing for mesh queries on size/position
   * \return The hydrostate defined at the given boundary position 
   */
  template<int ndim>
  KOKKOS_INLINE_FUNCTION
  ConsState getBoundaryValue(const typename ForeachCell::CellArray_global &Uin,
                             const typename ForeachCell::CellIndex        &iCell_boundary,
                             const typename ForeachCell::CellMetaData     &metadata) const 
  {
    // Retrieve symmetrical cell
    using CellIndex = typename ForeachCell::CellIndex;
    using offset_t =  typename CellIndex::offset_t;
    
    CellIndex iCell_ref;
    offset_t  offset;
    iCell_boundary.getBoundarySymmetrical(iCell_ref, offset);

    // By default, we define the result as the reference state (ie the symmetrical state wrt the boundary)
    ConsState res{}, ref{};
    getConservativeState<ndim>(Uin, iCell_ref, ref);
    res = ref;

    // X dir
    if (offset[IX] > 0) {
      if (bc_min[IX] == BC_REFLECTING)
        res.rho_u = -ref.rho_u;
      else if (bc_min[IX] == BC_USER)
        res = getUserdefBoundaryValue<ndim>(Uin, iCell_boundary, metadata, offset, IX);
    }
    else if (offset[IX] < 0) {
      if (bc_max[IX] == BC_REFLECTING)
        res.rho_u = -ref.rho_u;
      else if (bc_max[IX] == BC_USER)
        res = getUserdefBoundaryValue<ndim>(Uin, iCell_boundary, metadata, offset, IX);
    }

    // Y dir
    if (offset[IY] > 0) {
      if (bc_min[IY] == BC_REFLECTING)
        res.rho_v = -ref.rho_v;
      else if (bc_min[IY] == BC_USER)
        res = getUserdefBoundaryValue<ndim>(Uin, iCell_boundary, metadata, offset, IY);
    }
    else if (offset[IY] < 0) {
      if (bc_max[IY] == BC_REFLECTING)
        res.rho_v = -ref.rho_v;
      else if (bc_max[IY] == BC_USER)
        res = getUserdefBoundaryValue<ndim>(Uin, iCell_boundary, metadata, offset, IY);
    }

    // Z dir
    if (ndim == 3) {
      if (offset[IZ] > 0) {
        if (bc_min[IZ] == BC_REFLECTING)
          res.rho_w = -ref.rho_w;
        else if (bc_min[IZ] == BC_USER)
          res = getUserdefBoundaryValue<ndim>(Uin, iCell_boundary, metadata, offset, IZ);
      }
      else if (offset[IZ] < 0) {
        if (bc_max[IZ] == BC_REFLECTING)
          res.rho_w = -ref.rho_w;
        else if (bc_max[IZ] == BC_USER)
          res = getUserdefBoundaryValue<ndim>(Uin, iCell_boundary, metadata, offset, IZ);
      } 
    }
    

    return res;    
  }

  /**
   * \brief Returns a user-defined value at a boundary 
   * 
   * This method is called by getBoundaryValue in the case where we try to access
   * a value at the boundary that is flagged as user-defined. 
   * 
   * This method should return a conservative state.
   * 
   * \tparam ndim number of dimensions
   * 
   * \param Uin input array of conservative variables
   * \param iCell_boundary cell index outside of the domain (iCell_boundary.is_boundary() must be true)
   * \param metadata CellMetaData object allowing for mesh queries on size/position
   * \param offset the offset to apply to iCell_boundary to get to the symmetrical cell inside the domain
   * \param dir the direction along which we're getting out of the domain
   **/
  template <int ndim>
  KOKKOS_INLINE_FUNCTION
  ConsState getUserdefBoundaryValue(const typename ForeachCell::CellArray_global    &Uin, 
                                    const typename ForeachCell::CellIndex           &iCell_Boundary, 
                                    const typename ForeachCell::CellMetaData        &metadata, 
                                    const typename ForeachCell::CellIndex::offset_t &offset, 
                                    ComponentIndex3D                                 dir) const 
  {
    return ConsState {};
  }

  /**
   * \brief Overrides the flux at a boundary
   * 
   * Overrides a flux after the resolution of the Riemann problem at a boundary
   * 
   * \tparam ndim number of dimensions
   * 
   * \param flux_in the flux at the boundary as calculated per the Riemann solver
   * \param q the primitive value of the cell at the boundary, inside the domain
   * \param dir IX, IY or IZ depending on the direction of the current flux
   * \param min_bound tells if the current boundary is the minimum boundary along dir or the maximum
   * 
   **/
  template<int ndim>
  KOKKOS_INLINE_FUNCTION
  ConsState overrideBoundaryFlux(const ConsState flux_in, 
                                 const PrimState q,
                                 const ComponentIndex3D dir,
                                 const bool min_bound) const 
  {
    return flux_in;
  }

  /** Attributes **/
  RiemannParams params;

  BoundaryConditionType bc_min[3], bc_max[3];
  real_t xmin, xmax;
  real_t ymin, ymax;
  real_t zmin, zmax; 

};
}