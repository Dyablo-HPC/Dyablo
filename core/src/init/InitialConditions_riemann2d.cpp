#include "InitialConditions_analytical.h"

#include "AnalyticalFormula_tools.h"

namespace dyablo{

/**
 * 2D Riemann test.
 * Based on Lax, Liu "Solution of Two-Dimensional Riemann Problems of Gas Dynamics by Positive Schemes"
 * 1998, SIAM Journal on Scientific Computing
 **/
struct AnalyticalFormula_riemann2d : public AnalyticalFormula_base{
  
  const int    ndim;
  const real_t gamma0;
  const real_t smallr;
  const real_t smallc;
  const real_t smallp;
  const real_t error_max;

  const real_t x0, y0; // Position of the separation
  const int test_case;

  AnalyticalFormula_riemann2d( ConfigMap& configMap ) : 
    ndim(configMap.getValue<int>("mesh", "ndim", 2)),
    gamma0(configMap.getValue<real_t>("hydro","gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro","smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro","smallc", 1e-10)),
    smallp(smallc*smallc / gamma0),
    error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),
    
    x0(configMap.getValue<real_t>("riemann2d", "x0", 0.5)),
    y0(configMap.getValue<real_t>("riemann2d", "y0", 0.5)),
    test_case(configMap.getValue<int>("riemann2d", "test_case", 1)-1)
  {
    DYABLO_ASSERT_HOST_RELEASE( test_case >= 0 && test_case < 19, 
      "There are 19 riemann test cases, you used riemann2d/test_case = " << test_case);
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
    constexpr real_t pressures[19][4] = {{0.0439,0.1500,0.4000,1.0000},
                                        {1.0000,0.4000,0.4000,1.0000},
                                        {0.0290,0.3000,0.3000,1.5000},
                                        {1.1000,0.3500,0.3500,1.1000},
                                        {1.0000,1.0000,1.0000,1.0000},
                                        {1.0000,1.0000,1.0000,1.0000},
                                        {0.4000,0.4000,0.4000,1.0000},
                                        {1.0000,1.0000,1.0000,0.4000},
                                        {0.4000,0.4000,1.0000,1.0000},
                                        {0.3333,0.3333,1.0000,1.0000},
                                        {0.4000,0.4000,0.4000,1.0000},
                                        {1.0000,1.0000,1.0000,0.4000},
                                        {0.4000,0.4000,1.0000,1.0000},
                                        {2.6667,2.6667,8.0000,8.0000},
                                        {0.4000,0.4000,0.4000,1.0000},
                                        {1.0000,1.0000,1.0000,0.4000},
                                        {0.4000,0.4000,1.0000,1.0000},
                                        {0.4000,0.4000,1.0000,1.0000},
                                        {0.4000,0.4000,1.0000,1.0000}};

    constexpr real_t densities[19][4] = {{0.1072,0.2579,0.5197,1.0000},
                                        {1.0000,0.5197,0.5197,1.0000},
                                        {0.1380,0.5323,0.5323,1.5000},
                                        {1.1000,0.5065,0.5065,1.1000},
                                        {1.0000,3.0000,2.0000,1.0000},
                                        {1.0000,3.0000,2.0000,1.0000},
                                        {0.8000,0.5197,0.5197,1.0000},
                                        {0.8000,1.0000,1.0000,0.5197},
                                        {1.0390,0.5197,2.0000,1.0000},
                                        {0.2281,0.4562,0.5000,1.0000},
                                        {0.8000,0.5313,0.5313,1.0000},
                                        {0.8000,1.0000,1.0000,0.5313},
                                        {1.0625,0.5313,2.0000,1.0000},
                                        {0.4736,0.9474,1.0000,2.0000},
                                        {0.8000,0.5313,0.5197,1.0000},
                                        {0.8000,1.0000,1.0222,0.5313},
                                        {1.0625,0.5197,2.0000,1.0000},
                                        {1.0625,0.5197,2.0000,1.0000},
                                        {1.0625,0.5197,2.0000,1.0000}};

    constexpr real_t vx[19][4] = {{-0.7259,0.0000,-0.7259,0.0000},
                                  {-0.7259,0.0000,-0.7259,0.0000},
                                  {1.2060,0.0000,1.2060,0.0000},
                                  {0.8939,0.0000,0.8939,0.0000},
                                  {0.7500,0.7500,-0.7500,-0.7500},
                                  {-0.7500,-0.7500,0.7500,0.7500},
                                  {0.1000,0.1000,-0.6259,0.1000},
                                  {0.1000,0.1000,-0.6259,0.1000},
                                  {0.0000,0.0000,0.0000,0.0000},
                                  {0.0000,0.0000,0.0000,0.0000},
                                  {0.1000,0.1000,0.8276,0.1000},
                                  {0.0000,0.0000,0.7276,0.0000},
                                  {0.0000,0.0000,0.0000,0.0000},
                                  {0.0000,0.0000,0.0000,0.0000},
                                  {0.1000,0.1000,-0.6259,0.1000},
                                  {0.1000,0.1000,-0.6179,0.1000},
                                  {0.0000,0.0000,0.0000,0.0000},
                                  {0.0000,0.0000,0.0000,0.0000},
                                  {0.0000,0.0000,0.0000,0.0000}};

    constexpr real_t vy[19][4] = {{-1.4045,-1.4045,0.0000,0.0000},
                                  {-0.7259,-0.7259,0.0000,0.0000},
                                  {1.2060,1.2060,0.0000,0.0000},
                                  {0.8939,0.8939,0.0000,0.0000},
                                  {0.5000,-0.5000,0.5000,-0.5000},
                                  {0.5000,-0.5000,0.5000,-0.5000},
                                  {0.1000,-0.6259,0.1000,0.1000},
                                  {0.1000,-0.6259,0.1000,0.1000},
                                  {-0.8133,-0.4259,-0.3000,0.3000},
                                  {-0.6076,-0.4297,0.6076,0.4297},
                                  {0.0000,0.7276,0.0000,0.0000},
                                  {0.0000,0.7276,0.0000,0.0000},
                                  {0.8145,0.4276,0.3000,-0.3000},
                                  {1.2172,1.1606,-1.2172,-0.5606},
                                  {-0.3000,0.4276,-0.3000,-0.3000},
                                  {0.1000,0.8276,0.1000,0.1000},
                                  {0.2145,-1.1259,-0.3000,-0.4000},
                                  {0.2145,0.2741,-0.3000,1.0000},
                                  {0.2145,-0.4259,-0.3000,0.3000}};
    int quadrant;
    if (y < y0)
      quadrant = (x < x0 ? 0 : 1);
    else 
      quadrant = (x < x0 ? 2 : 3);

    ConsHydroState res;
    const real_t rho = densities[test_case][quadrant];
    const real_t u = vx[test_case][quadrant];
    const real_t v = vy[test_case][quadrant];
    const real_t Ek = 0.5 * rho * (u*u + v*v);
    res.rho   = rho;
    res.e_tot = Ek + pressures[test_case][quadrant] / (gamma0-1.0);
    res.rho_u = rho*u;
    res.rho_v = rho*v;

    return res;
  }
};
} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_riemann2d>, 
                 "riemann2d");

