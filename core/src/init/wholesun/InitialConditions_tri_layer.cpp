#include "../InitialConditions_analytical.h"

#include "../AnalyticalFormula_tools.h"

#include <Kokkos_Random.hpp>

namespace dyablo{

/**
 * @brief A three-layer setup very loosely based on the papers by Currie et al 2020, Brummel et al 2002 and Hurlburt et al 1996 
 * 
 * This setup takes two polytropic models (theta1, m1, kappa1) and (theta2, m2, kappa2) and initialises three layers.
 * The two outermost ones have use the second polytropic model while the central one uses the first polytropic model.
 * The idea is to have the first polytropic model to be convectively unstable while the two other stable.
 * The delimitation of the three layers is as follow :
 * 
 * zmin -- top stable layer -- z1 -- convective layer -- z2 -- bottom stable layer -- zmax
 * 
 */
struct AnalyticalFormula_tri_layer : public AnalyticalFormula_base{
  using RNGPool = Kokkos::Random_XorShift64_Pool<>;
  using RNGType = RNGPool::generator_type;

  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;

  const int    seed;            // Seed number for the random perturbation
  const real_t z1, z2;          // Position of the two transition layers
  const real_t theta1, theta2;  // Gradients in the convective and stable layers
  const real_t m1, m2;          // Polytropic indices of the convective and stable layers
  const real_t perturbation;    // Amplitude of the small random pressure perturbation

  RNGPool      rand_pool;  // Random pool for multi-mode perturbation

  AnalyticalFormula_tri_layer( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    seed(configMap.getValue<int>("tri_layer", "seed", 12345)),
    z1(configMap.getValue<real_t>("tri_layer", "z1", 1.0)),
    z2(configMap.getValue<real_t>("tri_layer", "z2", 1.0)),
    theta1(configMap.getValue<real_t>("tri_layer", "theta1", 1.0)),
    theta2(configMap.getValue<real_t>("tri_layer", "theta2", 1.0)),
    m1(configMap.getValue<real_t>("tri_layer", "m1", 1.0)),
    m2(configMap.getValue<real_t>("tri_layer", "m2", 1.0)),
    perturbation(configMap.getValue<real_t>("tri_layer", "perturbation", 1.0e-3)),
    rand_pool(seed*GlobalMpiSession::get_comm_world().MPI_Comm_rank()+1)

  {
  }

  KOKKOS_INLINE_FUNCTION
  bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const 
  {
    real_t gamma0 = this->gamma0;
    real_t smallr = this->smallr;
    real_t smallp = this->smallp;
    real_t error_max = this->error_max;
    return AnalyticalFormula_tools::auto_refine( *this, gamma0, smallr, smallp, error_max,
                                                  x, y, z, dx, dy, dz );
  }

  KOKKOS_INLINE_FUNCTION
  ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    // d is the depth. y in 2D, z in 3D
    real_t d = (ndim == 2 ? y : z);

    // Calculating the top-value for the three layers
    //  . subscript 0 is at d_min
    //  . subscript 1 is at z1
    //  . subscript 2 is at z2
    //
    // NOTE : We start at T0, rho0 = 10.0 to avoid instabilities.
    //        Since we are fully compressible, and not using any perturbation
    //        modelling with a background state, the pressure waves can be oscillating
    //        a lot and might lead to overshooting to negative pressures leading to 
    //        failures, code crash or NaNs production. 
    //        
    //        Empirically, T0/rho0=10.0 seem to work with Stiffnesses up to 7., for theta1 = 10.0

    const real_t T0 = 10.0;
    const real_t rho0 = 10.0;
    const real_t p0 = rho0*T0;

    const real_t T1   = T0 + theta2 * z1;
    const real_t rho1 = rho0 * pow(T1/T0, m2);
    const real_t p1   = p0 * pow(T1/T0, m2+1.0);

    const real_t T2   = T1 + theta1 * (z2-z1);
    const real_t rho2 = rho1 * pow(T2/T1, m1);
    const real_t p2   = p1 * pow(T2/T1, m1+1.0);

    // Top layer (stable)
    PrimHydroState q;
    real_t T;
    if (d <= z1) {
      T = T0 + theta2 * d;
      q.rho = rho0 * pow(T/T0, m2);
      q.p   = p0 * pow(T/T0, m2+1.0);
    }
    // Middle layer (convective)
    else if (d <= z2) {
      T = T1 + theta1* (d-z1);

      // We add a pressure perturbation as in C91
      RNGType rand_gen = rand_pool.get_state();
      real_t pert = perturbation * rand_gen.drand(-0.5, 0.5);
      rand_pool.free_state(rand_gen);
      
      q.rho = rho1 * pow(T/T1, m1);
      q.p   = p1 * (1.0 + pert) * pow(T/T1, m1+1.0);
    }
    // Bottom layer (stable)
    else {
      T = T2 + theta2 * (d-z2);
      q.rho = rho2 * pow(T/T2, m2);
      q.p   = p2 * pow(T/T2, m2+1.0);
    }

    q.u   = 0.0;
    q.v   = 0.0;
    q.w   = 0.0;

    ConsHydroState res = primToCons<3>(q, gamma0);
    return res; 
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_tri_layer>, "tri_layer");

