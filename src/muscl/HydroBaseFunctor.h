#ifndef HYDRO_BASE_FUNCTORS_H_
#define HYDRO_BASE_FUNCTORS_H_

#include <type_traits>

#include "shared/kokkos_shared.h"
#include "shared/HydroParams.h"
#include "shared/HydroState.h"

namespace euler_pablo { namespace muscl {

/**
 * SDM base functor, this is not a functor, but a base class to derive an actual
 * Kokkos functor.
 */
class HydroBaseFunctor
{
  
public:
  //! Decide at compile-time which HydroState to use
  //using HydroState = typename std::conditional<dim==2,HydroState2d,HydroState3d>::type;  
  
  HydroBaseFunctor(HydroParams params) :
    params(params) {};

  virtual ~HydroBaseFunctor() {};

  //! field indexes for velocity gradients computations (needed in viscous terms)

  ////// static constexpr are not supported by nvcc /////
  // enum grad_index_t {
  //   IGU  = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGU),
  //   IGV  = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGV),
  //   IGW  = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGW),
    
  //   IGUX = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGUX),
  //   IGUY = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGUY),
  //   IGUZ = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGUZ),
    
  //   IGVX = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGVX),
  //   IGVY = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGVY),
  //   IGVZ = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGVZ),

  //   IGWX = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGWX),
  //   IGWY = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGWY),
  //   IGWZ = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGWZ),
    
  //   IGT  = static_cast<int>(std::conditional<dim==2,VarIndexGrad2d,VarIndexGrad3d>::type::IGT),
  // };
  
  //! alias enum values used in EulerEquations flux computations
  ////// static constexpr are not supported by nvcc /////
  // enum flux_index_t {
  //   U_X  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::U_X),
  //   U_Y  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::U_Y),
  //   U_Z  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::U_Z),
    
  //   V_X  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::V_X),
  //   V_Y  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::V_Y),
  //   V_Z  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::V_Z),
    
  //   W_X  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::W_X),
  //   W_Y  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::W_Y),
  //   W_Z  = static_cast<int>(std::conditional<dim==2,gradientV_IDS_2d,gradientV_IDS_3d>::type::W_Z)
  // };
  
  HydroParams params;
  
  /**
   * a dummy swap device routine.
   */
  KOKKOS_INLINE_FUNCTION
  void swap ( real_t& a, real_t& b ) const {
    real_t c = a; a=b; b=c;
  }
  
  /**
   * Equation of state:
   * compute pressure p and speed of sound c, from density rho and
   * internal energy eint using the "calorically perfect gas" equation
   * of state : \f$ eint=\frac{p}{\rho (\gamma-1)} \f$
   * Recall that \f$ \gamma \f$ is equal to the ratio of specific heats
   *  \f$ \left[ c_p/c_v \right] \f$.
   * 
   * @param[in]  rho  density
   * @param[in]  eint internal energy
   * @param[out] p    pressure
   * @param[out] c    speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void eos(real_t rho,
	   real_t eint,
	   real_t* p,
	   real_t* c) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallp = params.settings.smallp;
    
    *p = FMAX((gamma0 - ONE_F) * rho * eint, rho * smallp);
    *c = SQRT(gamma0 * (*p) / rho);
    
  } // eos

  /**
   * Convert conservative variables (rho, rho*u, rho*v, e) to 
   * primitive variables (rho,u,v,p)
   * @param[in]  u  conservative variables array
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant nbvar)
   * @param[out] c  local speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void computePrimitives(const HydroState2d &u,
			 real_t             *c,
			 HydroState2d       &q) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    
    real_t d, p, ux, uy;
    
    d = fmax(u[ID], smallr);
    ux = u[IU] / d;
    uy = u[IV] / d;
    
    real_t eken = HALF_F * (ux*ux + uy*uy);
    real_t e = u[IP] / d - eken;
    
    // compute pressure and speed of sound
    p = fmax((gamma0 - 1.0) * d * e, d * smallp);
    *c = sqrt(gamma0 * (p) / d);
    
    q[ID] = d;
    q[IP] = p;
    q[IU] = ux;
    q[IV] = uy;
    
  } // computePrimitives - 2d

  /**
   * Convert conservative variables (rho, rho*u, rho*v, e) to 
   * primitive variables (rho,u,v,p)
   * @param[in]  u  conservative variables array
   * @param[out] q  primitive    variables array (allocated in calling routine, size is constant nbvar)
   * @param[out] c  local speed of sound
   */
  KOKKOS_INLINE_FUNCTION
  void computePrimitives(const HydroState3d &u,
			 real_t             *c,
			 HydroState3d       &q) const
  {
    real_t gamma0 = params.settings.gamma0;
    real_t smallr = params.settings.smallr;
    real_t smallp = params.settings.smallp;
    
    real_t d, p, ux, uy, uz;
    
    d = fmax(u[ID], smallr);
    ux = u[IU] / d;
    uy = u[IV] / d;
    uz = u[IW] / d;
    
    real_t eken = HALF_F * (ux*ux + uy*uy + uz*uz);
    real_t e = u[IP] / d - eken;
    
    // compute pressure and speed of sound
    p = fmax((gamma0 - 1.0) * d * e, d * smallp);
    *c = sqrt(gamma0 * (p) / d);
    
    q[ID] = d;
    q[IP] = p;
    q[IU] = ux;
    q[IV] = uy;
    q[IW] = uz;
    
  } // computePrimitives - 3d
  
}; // class HydroBaseFunctor

} // namespace muscl

} // namespace euler_pablo

#endif // HYDRO_BASE_FUNCTORS_H_
