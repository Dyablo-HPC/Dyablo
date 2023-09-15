#pragma once

#include "real_type.h"
#include "enums.h"
#include "kokkos_shared.h"
#include "State_Ops.h"
#include "State_MHD.h"
#include "FieldManager.h"

namespace dyablo {

/**
 * @brief Structure holding conservative magneto-hydrodynamics variables with divB cleaning
 **/ 
struct ConsGLMMHDState {
    enum VarIndex : dyablo::VarIndex
  {
    Irho,
    Ie_tot,
    Irho_vx,
    Irho_vy,
    Irho_vz,
    IBx, 
    IBy,
    IBz,
    Ipsi
  };

  static std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo()
  {
    return  { {"rho",     VarIndex::Irho}, 
              {"e_tot",   VarIndex::Ie_tot},
              {"rho_vx",  VarIndex::Irho_vx},
              {"rho_vy",  VarIndex::Irho_vy},
              {"rho_vz",  VarIndex::Irho_vz},
              {"Bx",      VarIndex::IBx},
              {"By",      VarIndex::IBy},
              {"Bz",      VarIndex::IBz},
              {"psi",     VarIndex::Ipsi} 
    };
  }
  
  static FieldManager getFieldManager()
  {
    return FieldManager( {
      VarIndex::Irho, 
      VarIndex::Ie_tot, 
      VarIndex::Irho_vx, 
      VarIndex::Irho_vy, 
      VarIndex::Irho_vz, 
      VarIndex::IBx, 
      VarIndex::IBy, 
      VarIndex::IBz, 
      VarIndex::Ipsi
    });
  }

  real_t rho = 0;
  real_t e_tot = 0;
  real_t rho_u = 0;
  real_t rho_v = 0;
  real_t rho_w = 0;
  real_t Bx = 0;
  real_t By = 0;
  real_t Bz = 0;
  real_t psi = 0;
};

DECLARE_STATE_TYPE( ConsGLMMHDState, 9 );
DECLARE_STATE_GET( ConsGLMMHDState, 0, rho );
DECLARE_STATE_GET( ConsGLMMHDState, 1, e_tot );
DECLARE_STATE_GET( ConsGLMMHDState, 2, rho_u );
DECLARE_STATE_GET( ConsGLMMHDState, 3, rho_v );
DECLARE_STATE_GET( ConsGLMMHDState, 4, rho_w );
DECLARE_STATE_GET( ConsGLMMHDState, 5, Bx );
DECLARE_STATE_GET( ConsGLMMHDState, 6, By );
DECLARE_STATE_GET( ConsGLMMHDState, 7, Bz );
DECLARE_STATE_GET( ConsGLMMHDState, 8, psi );

/**
 * @brief Structure holding primitive magneto-hydrodynamics variables
 */
struct PrimGLMMHDState {
  enum VarIndex : dyablo::VarIndex
  {
    Irho,
    Ip,
    Iu,
    Iv,
    Iw,
    IBx, 
    IBy,
    IBz,
    Ipsi
  }; 

  static std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo()
  {
    return  { {"rho",VarIndex::Irho}, 
              {"p",  VarIndex::Ip},
              {"u",  VarIndex::Iu},
              {"v",  VarIndex::Iv},
              {"w",  VarIndex::Iw},
              {"Bx", VarIndex::IBx},
              {"By", VarIndex::IBy},
              {"Bz", VarIndex::IBz},
              {"psi", VarIndex::Ipsi}
    };
  }

  static FieldManager getFieldManager()
  {
    return FieldManager( {
      VarIndex::Irho, 
      VarIndex::Ip, 
      VarIndex::Iu, 
      VarIndex::Iv, 
      VarIndex::Iw, 
      VarIndex::IBx, 
      VarIndex::IBy, 
      VarIndex::IBz, 
      VarIndex::Ipsi
    } );
  }


  real_t rho = 0;
  real_t p = 0;
  real_t u = 0;
  real_t v = 0;
  real_t w = 0;
  real_t Bx = 0;
  real_t By = 0;
  real_t Bz = 0;
  real_t psi = 0;
};

DECLARE_STATE_TYPE( PrimGLMMHDState, 9 );
DECLARE_STATE_GET( PrimGLMMHDState, 0, rho );
DECLARE_STATE_GET( PrimGLMMHDState, 1, p );
DECLARE_STATE_GET( PrimGLMMHDState, 2, u );
DECLARE_STATE_GET( PrimGLMMHDState, 3, v );
DECLARE_STATE_GET( PrimGLMMHDState, 4, w );
DECLARE_STATE_GET( PrimGLMMHDState, 5, Bx );
DECLARE_STATE_GET( PrimGLMMHDState, 6, By );
DECLARE_STATE_GET( PrimGLMMHDState, 7, Bz );
DECLARE_STATE_GET( PrimGLMMHDState, 8, psi );



/**
 * @brief Structure grouping the primitive and conservative MHD state as well
 *        as information on the number of fields to store per state
 */
struct GLMMHDState {
  using PrimState = PrimGLMMHDState;
  using ConsState = ConsGLMMHDState;
  static constexpr size_t N = 9;
};

/**
* @brief Returns a conservative state at a given cell index in an array
* 
* @tparam ndim the number of dimensions
* @tparam Array_t the type of array where we are looking up
* @tparam CellIndex the type of cell index used

* @param U the array in which we are getting the state
* @param iCell the index of the cell 
* @return the hydro state at position iCell in U
*/
template< int ndim, 
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getConservativeState( const Array_t& U, const CellIndex& iCell, ConsGLMMHDState &res )
{
  res.rho   = U.at(iCell, ConsGLMMHDState::VarIndex::Irho);
  res.e_tot = U.at(iCell, ConsGLMMHDState::VarIndex::Ie_tot);
  res.rho_u = U.at(iCell, ConsGLMMHDState::VarIndex::Irho_vx);
  res.rho_v = U.at(iCell, ConsGLMMHDState::VarIndex::Irho_vy);
  res.rho_w = (ndim == 3 ? U.at(iCell, ConsGLMMHDState::VarIndex::Irho_vz) : 0.0);
  res.Bx = U.at(iCell, ConsGLMMHDState::VarIndex::IBx);
  res.By = U.at(iCell, ConsGLMMHDState::VarIndex::IBy);
  res.Bz = U.at(iCell, ConsGLMMHDState::VarIndex::IBz);
  res.psi = U.at(iCell, ConsGLMMHDState::VarIndex::Ipsi);
}

/**
 * @brief Returns a primitive hydro state at a given cell index in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array where we are looking up
 * @tparam CellIndex the type of cell index used
 *
 * @param U the array in which we are getting the state
 * @param iCell the index of the cell 
 * @return the hydro state at position iCell in U
 */
template< int ndim,
          typename Array_t, 
          typename CellIndex >
KOKKOS_INLINE_FUNCTION
void getPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimGLMMHDState &res )
{
  res.rho = U.at(iCell, PrimGLMMHDState::VarIndex::Irho);
  res.p   = U.at(iCell, PrimGLMMHDState::VarIndex::Ip);
  res.u   = U.at(iCell, PrimGLMMHDState::VarIndex::Iu);
  res.v   = U.at(iCell, PrimGLMMHDState::VarIndex::Iv);
  res.w   = (ndim == 3 ? U.at(iCell, PrimGLMMHDState::VarIndex::Iw) : 0.0);
  res.Bx  = U.at(iCell, PrimGLMMHDState::VarIndex::IBx);
  res.By  = U.at(iCell, PrimGLMMHDState::VarIndex::IBy);
  res.Bz  = U.at(iCell, PrimGLMMHDState::VarIndex::IBz);
  res.psi = U.at(iCell, PrimGLMMHDState::VarIndex::Ipsi);
}

/**
 * @brief Stores a primitive hydro state in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array in which the primitive value is stored
 * @tparam CellIndex the type of cell index used
 * 
 * @param U the array where we are storing the state
 * @param iCell the index of cell
 * @param u the value to store in the array
 */
template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setPrimitiveState( const Array_t& U, const CellIndex& iCell, PrimGLMMHDState u) {
  U.at(iCell, PrimGLMMHDState::VarIndex::Irho) = u.rho;
  U.at(iCell, PrimGLMMHDState::VarIndex::Ip) = u.p;
  U.at(iCell, PrimGLMMHDState::VarIndex::Iu) = u.u;
  U.at(iCell, PrimGLMMHDState::VarIndex::Iv) = u.v;
  U.at(iCell, PrimGLMMHDState::VarIndex::IBx) = u.Bx;
  U.at(iCell, PrimGLMMHDState::VarIndex::IBy) = u.By;
  U.at(iCell, PrimGLMMHDState::VarIndex::IBz) = u.Bz;
  if (ndim == 3) {
    U.at(iCell, PrimGLMMHDState::VarIndex::Iw) = u.w;
  }
  U.at(iCell, PrimGLMMHDState::VarIndex::Ipsi) = u.psi;
}

/**
 * @brief Stores a conservative hydro state in an array
 * 
 * @tparam ndim the number of dimensions
 * @tparam Array_t the type of array in which the primitive value is stored
 * @tparam CellIndex the type of cell index used
 * 
 * @param U the array where we are storing the state
 * @param iCell the index of cell
 * @param u the value to store in the array
 */
template <int ndim, typename Array_t, typename CellIndex >
KOKKOS_INLINE_FUNCTION
void setConservativeState( const Array_t& U, const CellIndex& iCell, ConsGLMMHDState u) {
  U.at(iCell, ConsGLMMHDState::VarIndex::Irho) = u.rho;
  U.at(iCell, ConsGLMMHDState::VarIndex::Ie_tot) = u.e_tot;
  U.at(iCell, ConsGLMMHDState::VarIndex::Irho_vx) = u.rho_u;
  U.at(iCell, ConsGLMMHDState::VarIndex::Irho_vy) = u.rho_v;
  U.at(iCell, ConsGLMMHDState::VarIndex::IBx) = u.Bx;
  U.at(iCell, ConsGLMMHDState::VarIndex::IBy) = u.By;
  U.at(iCell, ConsGLMMHDState::VarIndex::IBz) = u.Bz;
  if (ndim == 3) {
    U.at(iCell, ConsGLMMHDState::VarIndex::Irho_vz) = u.rho_w;
  }
  U.at(iCell, ConsGLMMHDState::VarIndex::Ipsi) = u.psi;
}

/**
 * @brief Converts from a MHD conservative state to a MHD primitive state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param U the initial conservative state
 * @param gamma0 adiabatic index
 * @return the primitive version of U
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
PrimGLMMHDState consToPrim(const ConsGLMMHDState &U, real_t gamma0) {
  const real_t Ek = 0.5 * (U.rho_u*U.rho_u
                          +U.rho_v*U.rho_v
                          +(ndim == 3 ? U.rho_w*U.rho_w : 0.0))/U.rho;

  const real_t Em = 0.5 * (U.Bx*U.Bx + U.By*U.By + U.Bz*U.Bz);
  const real_t p = (U.e_tot - Ek - Em) * (gamma0-1.0);
  return {U.rho, 
          p, 
          U.rho_u/U.rho, 
          U.rho_v/U.rho, 
          (ndim == 3 ? U.rho_w/U.rho : 0.0),
          U.Bx,
          U.By,
          U.Bz,
          U.psi};
}

/**
 * @brief Converts from a MHD primitive state to a MHD conservative state
 * 
 * @tparam ndim the number of dimensions
 * 
 * @param Q the initial primitive state
 * @param gamma0 adiabatic index
 * @return the conservative version of Q
 */
template<int ndim>
KOKKOS_INLINE_FUNCTION
ConsGLMMHDState primToCons(const PrimGLMMHDState &Q, real_t gamma0) {
  const real_t Ek = 0.5 * Q.rho * (Q.u*Q.u
                                  +Q.v*Q.v
                                  +(ndim == 3 ? Q.w*Q.w : 0.0));
                                  
  const real_t Em = 0.5 * (Q.Bx*Q.Bx + Q.By*Q.By + Q.Bz*Q.Bz);
  const real_t E  = Ek + Em + Q.p / (gamma0-1.0);
  return {Q.rho, 
          E, 
          Q.rho*Q.u, 
          Q.rho*Q.v, 
          (ndim ==3 ? Q.rho*Q.w : 0.0),
          Q.Bx,
          Q.By,
          Q.Bz,
          Q.psi};
}

/**
 * @brief Swaps a component in velocity and magnetic field with the X component. 
 *        The Riemann problem is always solved by considering an interface on the 
 *        X-axis. So when solving it for other components, those should be swapped 
 *        before and after solving the Riemann problem.
 *  
 * @param Q (IN/OUT) the primitive MHD state to modify
 * @param comp the component to swap with X
 */
KOKKOS_INLINE_FUNCTION
PrimGLMMHDState swapComponents(const PrimGLMMHDState &q, ComponentIndex3D comp) {
  switch( comp )
  {
    case IX:
      return q;
    case IY:
      return PrimGLMMHDState{q.rho, q.p, q.v, q.u, q.w, q.By, q.Bx, q.Bz, q.psi};
    case IZ:
      return PrimGLMMHDState{q.rho, q.p, q.w, q.v, q.u, q.Bz, q.By, q.Bx, q.psi};
    default:
      assert(false);
      return PrimGLMMHDState{};
  }
}

/**
 * @brief Swaps a component in velocity and magnetic field with the X component. 
 *        The Riemann problem is always solved by considering an interface on the 
 *        X-axis. So when solving it for other components, those should be swapped 
 *        before and after solving the Riemann problem.
 *  
 * @param Q (IN/OUT) the primitive MHD state to modify
 * @param comp the component to swap with X
 */
KOKKOS_INLINE_FUNCTION
ConsGLMMHDState swapComponents(const ConsGLMMHDState &u, ComponentIndex3D comp) {
  switch( comp )
  {
    case IX:
      return u;
    case IY:
      return ConsGLMMHDState{u.rho, u.e_tot, u.rho_v, u.rho_u, u.rho_w, u.By, u.Bx, u.Bz, u.psi};
    case IZ:
      return ConsGLMMHDState{u.rho, u.e_tot, u.rho_w, u.rho_v, u.rho_u, u.Bz, u.By, u.Bx, u.psi};
    default:
      assert(false);
      return ConsGLMMHDState{};
  }
}

/**
 * @brief Converts from GLM MHD state to simple MHD state (without psi)
 */
KOKKOS_INLINE_FUNCTION
PrimMHDState GLMToMHD(const PrimGLMMHDState &q) {
  return PrimMHDState {q.rho, q.p, q.u, q.v, q.w, q.Bx, q.By, q.Bz};
}

} // namespace dyablo


