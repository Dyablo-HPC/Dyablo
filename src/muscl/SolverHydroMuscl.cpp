#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "muscl/SolverHydroMuscl.h"
#include "shared/HydroParams.h"

//#include "shared/mpiBorderUtils.h"

namespace euler_pablo { namespace muscl {

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void SolverHydroMuscl::make_boundaries(DataArray Udata)
{

  bool mhd_enabled = false;

#ifdef USE_MPI

  make_boundaries_mpi(Udata, mhd_enabled);

#else

  make_boundaries_serial(Udata, mhd_enabled);
  
#endif // USE_MPI
  
} // SolverHydroMuscl::make_boundaries

// =======================================================
// =======================================================
/**
 * Hydrodynamical Implosion Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/implode/Implode.html
 */
void SolverHydroMuscl::init_implode(DataArray Udata)
{


} // SolverHydroMuscl::init_implode

// =======================================================
// =======================================================
/**
 * Hydrodynamical blast Test.
 * http://www.astro.princeton.edu/~jstone/Athena/tests/blast/blast.html
 */
void SolverHydroMuscl::init_blast(DataArray Udata)
{


} // SolverHydroMuscl::init_blast

// =======================================================
// =======================================================
/**
 * Hydrodynamical Kelvin-Helmholtz instability Test.
 *
 * see https://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
 *
 * See also article by Robertson et al:
 * "Computational Eulerian hydrodynamics and Galilean invariance", 
 * B.E. Robertson et al, Mon. Not. R. Astron. Soc., 401, 2463-2476, (2010).
 *
 */
void SolverHydroMuscl::init_kelvin_helmholtz(DataArray Udata)
{


} // SolverHydroMuscl::init_kelvin_helmholtz

// =======================================================
// =======================================================
/**
 * Hydrodynamical Gresho vortex Test.
 *
 * \sa https://www.cfd-online.com/Wiki/Gresho_vortex
 * \sa https://arxiv.org/abs/1409.7395 - section 4.2.3
 * \sa https://arxiv.org/abs/1612.03910
 *
 */
void SolverHydroMuscl::init_gresho_vortex(DataArray Udata)
{

} // SolverHydroMuscl::init_gresho_vortex

// =======================================================
// =======================================================
/**
 * Init four quadrant (piecewise constant).
 *
 * Four quadrant 2D riemann problem.
 *
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
void SolverHydroMuscl::init_four_quadrant(DataArray Udata)
{

  // int configNumber = configMap.getInteger("riemann2d","config_number",0);
  // real_t xt = configMap.getFloat("riemann2d","x",0.8);
  // real_t yt = configMap.getFloat("riemann2d","y",0.8);

  // HydroState2d U0, U1, U2, U3;
  // getRiemannConfig2d(configNumber, U0, U1, U2, U3);
  
  // primToCons_2D(U0, params.settings.gamma0);
  // primToCons_2D(U1, params.settings.gamma0);
  // primToCons_2D(U2, params.settings.gamma0);
  // primToCons_2D(U3, params.settings.gamma0);

  // InitFourQuadrantFunctor2D::apply(params, Udata, configNumber,
  // 				   U0, U1, U2, U3,
  // 				   xt, yt, nbCells);
  
} // SolverHydroMuscl::init_four_quadrant


// =======================================================
// =======================================================
/**
 * Isentropic vortex advection test.
 * https://www.cfd-online.com/Wiki/2-D_vortex_in_isentropic_flow
 * https://hal.archives-ouvertes.fr/hal-01485587/document
 */
void SolverHydroMuscl::init_isentropic_vortex(DataArray Udata)
{
  
  //IsentropicVortexParams iparams(configMap);
  
  //InitIsentropicVortexFunctor::apply(params, iparams, Udata, nbCells);
  
} // SolverHydroMuscl::init_isentropic_vortex

// =======================================================
// =======================================================
void SolverHydroMuscl::init_rayleigh_taylor(DataArray Udata,
					    DataArray gravity)
{
  
  
} // SolverHydroMuscl::init_rayleigh_taylor

// =======================================================
// =======================================================
/**
 * Hydrodynamical rising bubble test.
 *
 */
void SolverHydroMuscl::init_rising_bubble(DataArray Udata,
					  DataArray gravity)
{
  
} // SolverHydroMuscl::init_rising_bubble

// =======================================================
// =======================================================
/**
 * Disk setup.
 *
 */
void SolverHydroMuscl::init_disk(DataArray Udata,
				 DataArray gravity)
{
  
  
} // SolverHydroMuscl::init_disk

// =======================================================
// =======================================================
void SolverHydroMuscl::init_restart(DataArray Udata)
{

  
} // SolverHydroMuscl::init_restart

// =======================================================
// =======================================================
void SolverHydroMuscl::init(DataArray Udata)
{

  // test if we are performing a re-start run (default : false)
  bool restartEnabled = configMap.getBool("run","restart_enabled",false);

  if (restartEnabled) { // load data from input data file

    init_restart(Udata);
    
  } else { // regular initialization

    /*
     * initialize hydro array at t=0
     */
    if ( !m_problem_name.compare("implode") ) {
      
      init_implode(Udata);
      
    } else if ( !m_problem_name.compare("blast") ) {
      
      init_blast(Udata);
      
    } else if ( !m_problem_name.compare("kelvin_helmholtz") ) {
      
      init_kelvin_helmholtz(Udata);
      
    } else if ( !m_problem_name.compare("gresho_vortex") ) {
      
      init_gresho_vortex(Udata);
      
    } else if ( !m_problem_name.compare("four_quadrant") ) {
      
      init_four_quadrant(Udata);
      
    } else if ( !m_problem_name.compare("isentropic_vortex") ) {
      
      init_isentropic_vortex(Udata);
      
    } else if ( !m_problem_name.compare("rayleigh_taylor") ) {
      
      init_rayleigh_taylor(Udata,gravity);
      
    } else if ( !m_problem_name.compare("rising_bubble") ) {
      
      init_rising_bubble(Udata,gravity);
      
    } else if ( !m_problem_name.compare("disk") ) {
      
      init_disk(Udata,gravity);
      
    } else {
      
      std::cout << "Problem : " << m_problem_name
		<< " is not recognized / implemented."
		<< std::endl;
      std::cout <<  "Use default - implode" << std::endl;
      init_implode(Udata);
      
    }

  } // end regular initialization

} // SolverHydroMuscl::init

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme
// ///////////////////////////////////////////
void SolverHydroMuscl::godunov_unsplit_impl(DataArray data_in, 
					    DataArray data_out, 
					    real_t dt)
{
  
  // fill ghost cell in data_in
  m_timers[TIMER_BOUNDARIES]->start();
  make_boundaries(data_in);
  m_timers[TIMER_BOUNDARIES]->stop();
    
  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  // start main computation
  m_timers[TIMER_NUM_SCHEME]->start();

  // convert conservative variable into primitives ones for the entire domain
  convertToPrimitives(data_in);

  if (params.implementationVersion == 0) {
    
    // compute fluxes (if gravity_enabled is false,
    // the last parameter is not used)
    // ComputeAndStoreFluxesFunctor::apply(params, Q,
    // 					Fluxes_x, Fluxes_y,
    // 					dt,
    // 					m_gravity_enabled,
    // 					gravity);
    
    // actual update
    // UpdateFunctor::apply(params, data_out,
    // 			 Fluxes_x, Fluxes_y);    

    
  } else if (params.implementationVersion == 1) {

  } // end params.implementationVersion == 1
  
  m_timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMuscl::godunov_unsplit_impl


} // namespace muscl

} // namespace euler_pablo
