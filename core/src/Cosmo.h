#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace dyablo {

namespace Impl { 
namespace { 

real_t lookup_interpolate( const std::vector<real_t>& lookup_x, const std::vector<real_t>& lookup_y, real_t x )
  {
    auto lower_bound_it = std::lower_bound( lookup_x.begin(), lookup_x.end(), x );
    int i = lower_bound_it - lookup_x.begin(); // First greater or equal

    real_t x_interp = ( x - lookup_x[i-1] ) / ( lookup_x[i] - lookup_x[i-1] );
    return lookup_y[i-1] + x_interp * ( lookup_y[i] - lookup_y[i-1]  );
  }

/***
 * Integrate function f in [a,b] with accuracy 'acc' and steps 'max_steps'
 * using Romberg's method
 * https://en.wikipedia.org/wiki/Romberg%27s_method
 * @param f : real_t -> real_t function to integrate
 * @param a : lower limit
 * @param b : upper limit
 * @param max_steps: maximum steps of the procedure
 * @param acc  : desired accuracy
 ***/
template< typename Func /* real_t -> real_t */ >
real_t romberg(const Func& f, real_t a, real_t b, size_t max_steps, real_t acc) 
{
  real_t R1[max_steps], R2[max_steps]; // buffers
  real_t *Rp = &R1[0], *Rc = &R2[0]; // Rp is previous row, Rc is current row
  real_t h = b-a; //step size
  Rp[0] = (f(a) + f(b))*h*0.5; // first trapezoidal step

  for (size_t i = 1; i < max_steps; ++i) {
    h /= 2.;
    real_t c = 0;
    size_t ep = 1 << (i-1); //2^(n-1)
    for (size_t j = 1; j <= ep; ++j) {
      c += f(a + (2*j-1) * h);
    }
    Rc[0] = h*c + .5*Rp[0]; // R(i,0)

    for (size_t j = 1; j <= i; ++j) {
      real_t n_k = pow(4, j);
      Rc[j] = (n_k*Rc[j-1] - Rp[j-1]) / (n_k-1); // compute R(i,j)
    }

    if (i > 1 && fabs(Rp[i-1]-Rc[i]) < acc) {
      return Rc[i];
    }

    // swap Rn and Rc as we only need the last row
    real_t *rt = Rp;
    Rp = Rc;
    Rc = rt;
  }

  std::cout << "WARNING : Romberg did not converge in " << max_steps << " iterations; Error = " << fabs(Rp[max_steps-1]-Rc[max_steps]) << std::endl;
  return Rp[max_steps-1]; // return our best guess
}
  
} // namespace
} // namespace Impl::

/**
 * @brief Utility struct for all of the cosmo
 * 
 */
class CosmoManager {
private:
  static real_t integrate_da_dt(real_t omega_m, real_t omega_v, real_t a, real_t b, real_t tol)
  {
    constexpr int max_iterations=16;
    auto faexp_tilde = [&](real_t aexp)
    {
      return 1.0 / sqrt(std::pow(aexp, 4) * (1.0-omega_m-omega_v) 
                    + std::pow(aexp, 3) * omega_m 
                    + std::pow(aexp, 5) * omega_v);
    };
    return Impl::romberg( faexp_tilde, a, b, max_iterations, tol );
  }

public:
  CosmoManager(ConfigMap &configMap)
    : cosmo_run(configMap.getValue<bool>("cosmology", "active", false)),
      omega_m(configMap.getValue<real_t>("cosmology",  "omega_m", 0.3)),
      omega_v(configMap.getValue<real_t>("cosmology",  "omega_v", 0.7)),
      a_start(configMap.getValue<real_t>("cosmology",  "aStart",  1.0e-2)),
      a_end(configMap.getValue<real_t>("cosmology", "aEnd", 1.00)),
      da(configMap.getValue<real_t>("cosmology", "da", 1.02)),
      save_expansion_table(configMap.getValue<bool>("cosmology", "save_expansion_table", false)),
      lookup_size(configMap.getValue<size_t>("cosmology", "lookup_size", 1024)) {
    computeFLM();
  }

  real_t timeToExpansion(const real_t time) const 
  {   
    return Impl::lookup_interpolate( lookup_t, lookup_a, time );
  }  

  real_t expansionToTime(const real_t aexp) const 
  {
    return Impl::lookup_interpolate( lookup_a, lookup_t, aexp );
  }

  void computeFLM() {
    const real_t a_ext = a_end * 1.1; // Getting a safety margin
    const real_t delta_a = a_ext - a_start;
    
    lookup_a.reserve(lookup_size);
    lookup_t.reserve(lookup_size);
    for (size_t i=0; i < lookup_size; ++i) {
      real_t a = a_start + i*delta_a / (lookup_size-1);
      real_t t = -0.5 * sqrt(omega_m) * integrate_da_dt(omega_m, omega_v, a, 1.0, 1.0e-8);
      lookup_a.push_back(a);
      lookup_t.push_back(t);
    }

    if (save_expansion_table) 
    {
      std::ofstream f_out("expansion_table.dat");
      f_out << "#a t" << std::endl;
      for (size_t i=0; i < lookup_size; ++i)
      {
        f_out << lookup_a[i] << " " << lookup_t[i] << std::endl;
      }
    }
  }

  static real_t static_compute_cosmo_dt(real_t omega_m, real_t omega_v, real_t a, real_t da) {
    return -0.5 * sqrt(omega_m) * integrate_da_dt(omega_m, omega_v, a*da, a, 1.0e-8);
  }

  real_t compute_cosmo_dt(real_t a)
  {
    return static_compute_cosmo_dt(omega_m, omega_v, a, da);
  }


  // Members
  bool cosmo_run;          //!< Is the current run a cosmo run
  real_t omega_m, omega_v; //!< Energy budget
  real_t a_start, a_end;   //!< Expansion factor at the start and at the end of the simulation
  real_t da;               //!< By how much do we need to multiplpy a for the next step

  bool save_expansion_table;

  size_t lookup_size;
  std::vector<real_t> lookup_a;
  std::vector<real_t> lookup_t;
};

}