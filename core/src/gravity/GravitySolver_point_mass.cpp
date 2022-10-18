#include "GravitySolver_analytical.tpp"

namespace dyablo{

/**
 * AnalyticalFormula_gravity implementation for point mass
 * to use with GravitySolver_analytical
 **/
class AnalyticalFormula_gravity_point_mass : public AnalyticalFormula_gravity
{
private:
    
  // Position of the central mass
  real_t point_mass_x, point_mass_y, point_mass_z; 
  // Gravity parameter of the central mass (gravitational constant G times the object mass)
  real_t point_mass_Gmass;
  // Gravitational softening, to avoid divergence of the gravitational force.
  real_t softening;                                


public:
  AnalyticalFormula_gravity_point_mass( ConfigMap& configMap )
  : point_mass_x( configMap.getValue<real_t>("gravity", "point_mass_x",  0.5) ),
    point_mass_y( configMap.getValue<real_t>("gravity", "point_mass_y",  0.5) ),
    point_mass_z( configMap.getValue<real_t>("gravity", "point_mass_z",  0.5) ),
    point_mass_Gmass( configMap.getValue<real_t>("gravity", "point_mass_Gmass", 1) ),
    softening( configMap.getValue<real_t>("gravity", "softening",  0.005) )
  {}

  KOKKOS_INLINE_FUNCTION
  GravityField_t value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    // Centered coordinates
    real_t rx = x - point_mass_x;
    real_t ry = y - point_mass_y;
    real_t rz = z - point_mass_z;

    // Square radius from central object (softened)
    real_t r2 = rx*rx + ry*ry + rz*rz + softening*softening;

    real_t r = sqrt(r2);
    real_t const_term = - point_mass_Gmass / (r2 * r);

    return{
      const_term * rx, // gx
      const_term * ry, // gy
      const_term * rz  // gz (this will not be written in 2D)
    };
  }
};

} // namespace dyablo

FACTORY_REGISTER( dyablo::GravitySolverFactory, 
                  dyablo::GravitySolver_analytical<dyablo::AnalyticalFormula_gravity_point_mass>, 
                  "GravitySolver_point_mass" );