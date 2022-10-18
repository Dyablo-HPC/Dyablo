#include "GravitySolver_analytical.tpp"

namespace dyablo{

/**
 * AnalyticalFormula_gravity implementation for point mass
 * to use with GravitySolver_analytical
 **/
class AnalyticalFormula_gravity_point_mass : public AnalyticalFormula_gravity
{
private:
  real_t star_x0, star_y0, star_z0;
  real_t star_mass, softening;


public:
  AnalyticalFormula_gravity_point_mass( ConfigMap& configMap )
  : star_x0( configMap.getValue<real_t>("gravity", "star_x0",  0.5) ),
    star_y0( configMap.getValue<real_t>("gravity", "star_y0",  0.5) ),
    star_z0( configMap.getValue<real_t>("gravity", "star_z0",  0.5) ),
    star_mass( configMap.getValue<real_t>("gravity", "star_mass", 1) ),
    softening( configMap.getValue<real_t>("gravity", "softening",  0.005) )
  {}

  KOKKOS_INLINE_FUNCTION
  GravityField_t value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    // Centered coordinates
    real_t rx = x - star_x0;
    real_t ry = y - star_y0;
    real_t rz = z - star_z0;

    // Square radius from central object (softened)
    real_t r2 = rx*rx + ry*ry + rz*rz + softening*softening;

    real_t r = sqrt(r2);
    constexpr real_t G = 1.0 / (4.0 * M_PI); // Gravitational constant
    real_t const_term = - G * star_mass / (r2 * r);

    return{
      const_term * rx,
      const_term * ry,
      const_term * rz // This will not be written in 2D
    };
  }
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::GravitySolverFactory, 
                  dyablo::GravitySolver_analytical<dyablo::AnalyticalFormula_gravity_point_mass>, 
                  "GravitySolver_point_mass" );