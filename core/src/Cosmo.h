#pragma once

#include <string>
#include <vector>

namespace dyablo {

/**
 * @brief Utility struct for all of the cosmo
 * 
 */
class CosmoManager {
private:
  real_t faexp_tilde(real_t a) {
    return 1.0 / sqrt(std::pow(a, 4) * (1.0-omega_m-omega_v) 
                    + std::pow(a, 3) * omega_m 
                    + std::pow(a, 5) * omega_v);
  };

  real_t integrate_da_dt(real_t a, real_t b, real_t tol) {
    real_t h  = 0.5*(b-a);
    real_t gmax = h*(faexp_tilde(a) + faexp_tilde(b));
    int jmax;
    constexpr int max_tab=5;
    constexpr int max_iterations=16;

    std::vector<real_t> g(max_tab+2);
    g[1] = gmax;
    
    int N = 1;
    real_t err = 1.0e20;
    int nite = 0;
    real_t g0 = 0.0;

    while (fabs(err) > tol && nite < max_iterations) {
      nite++;
      
      g0 = 0.0;

      for (int k=1; k <= N; ++k)
        g0 += faexp_tilde(a + (2*k-1)*h);

      g0 = 0.5 * g[1] + h*g0;
      h  = 0.5*h;
      N *=2;
      jmax = (nite < max_tab ? nite : max_tab);
      real_t j4 = 1.0;

      for (int j=1; j <= jmax; ++j) {
        j4 *= 4;
        real_t g1 = g0 + (g0-g[j]) / (j4-1.0);
        g[j] = g0;
        g0 = g1;
      }

      if (fabs(g0) > tol)
        err = 1.0 - gmax/g0;
      else
        err = gmax;

      gmax = g0;
      g[jmax + 1] = g0;
    }

    if (nite >= max_iterations && (fabs(err) > tol))
      std::cout << "WARNING : Friedmann-Lemaître did not converge in " << nite << " iterations; Error = " << fabs(err) << std::endl;

    return g0;
  }

public:
  CosmoManager(ConfigMap &configMap)
    : cosmo_run(configMap.getValue<bool>("cosmology", "active", false)),
      omega_m(configMap.getValue<real_t>("cosmology",  "omega_m", 0.3)),
      omega_v(configMap.getValue<real_t>("cosmology",  "omega_v", 0.7)),
      a_start(configMap.getValue<real_t>("cosmology",  "aStart",  1.0e-2)),
      a_end(configMap.getValue<real_t>("cosmology", "aEnd", 1.00)),
      da(configMap.getValue<real_t>("cosmology", "da", 1.02)),
      levelMin(configMap.getValue<int>("amr", "level_min", 5)),
      bx(configMap.getValue<int>("amr", "bx", 1.0)),
      omega_b(configMap.getValue<real_t>("cosmology", "omegab", 1.0)),
      save_expansion_table(configMap.getValue<bool>("cosmology", "save_expansion_table", false)),
      lookup_size(configMap.getValue<size_t>("cosmology", "lookup_size", 1024)) {
    computeFLM();
  }

  bool isRunFinished(const real_t time) const {
    return timeToExpansion(time) >= a_end - 1.0e-14;
  }

  real_t timeToExpansion(const real_t time) const {
    int i=0;
    while (lookup_t[i] < time)
      i++;
    
    const real_t x = (time - lookup_t[i-1]) / (lookup_t[i]-lookup_t[i-1]);
    return (1.0-x) * lookup_a[i-1] + x * lookup_a[i];
  }

  real_t expansionToTime(const real_t aexp) const {
    int i=0;
    while (lookup_a[i] < aexp)
      i++;
    
    const real_t x = (aexp - lookup_a[i-1]) / (lookup_a[i]-lookup_a[i-1]);
    return (1.0-x) * lookup_t[i-1] + x * lookup_t[i];
  }

  void computeFLM() {
    const real_t a_ext = a_end * 1.1; // Getting a safety margin
    const real_t da = a_ext - a_start;
    std::ofstream f_out;
    if (save_expansion_table) {
      f_out.open("expansion_table.dat");
      f_out << "#a t" << std::endl;
    }

    for (int i=0; i < lookup_size; ++i) {
      real_t a = a_start + i*da / (lookup_size-1);
      real_t t = -0.5 * sqrt(omega_m) * integrate_da_dt(a, 1.0, 1.0e-8);
      lookup_a.push_back(a);
      lookup_t.push_back(t);

      if (save_expansion_table)
        f_out << a << " " << t << std::endl;
    }
    if (save_expansion_table)
      f_out.close();
  }

  real_t compute_cosmo_dt(const real_t a) {
    return -0.5 * sqrt(omega_m) * integrate_da_dt(a*da, a, 1.0e-8);
  }


  // Members
  bool cosmo_run;          //!< Is the current run a cosmo run
  real_t omega_m, omega_v; //!< Energy budget
  real_t a_start, a_end;   //!< Expansion factor at the start and at the end of the simulation
  real_t da;               //!< By how much do we need to multiplpy a for the next step
  int levelMin;
  int bx;
  const std::string graficDir;
  real_t omega_b;

  bool save_expansion_table;

  size_t lookup_size;
  std::vector<real_t> lookup_a;
  std::vector<real_t> lookup_t;
};

}