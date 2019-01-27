#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "shared/HydroParams.h"

#include "shared/SimpleVTKIO.h"

#include "muscl/SolverHydroMuscl.h"

// Init conditions functors
#include "muscl/init/HydroInitFunctors.h"

//#include "shared/mpiBorderUtils.h"

namespace euler_pablo { namespace muscl {

// =======================================================
// ==== CLASS SolverHydroMuscl IMPL ======================
// =======================================================

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl::SolverHydroMuscl(HydroParams& params,
				   ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), U2(), Q(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  Slopes_x(), Slopes_y(), Slopes_z()
{
  
  solver_type = SOLVER_MUSCL_HANCOCK;
  
  // m_nCells = nbCells; // TODO
  m_nDofsPerCell = 1;

  int nbvar = params.nbvar;
 
  long long int total_mem_size = 0;

  /*
   * memory pre-allocation.
   *
   * Note that Uhost is not just a view to U, Uhost will be used
   * to save data from multiple other device array.
   * That's why we didn't use create_mirror_view to initialize Uhost.
   */

  // minimal number of cells
  uint64_t nbCells = 1<<params.level_min;
  
  nbCells = params.dimType == TWO_D ? nbCells * nbCells :  nbCells * nbCells * nbCells;
  
  U     = DataArray("U", nbCells, nbvar);
  Uhost = Kokkos::create_mirror(U);
  U2    = DataArray("U2",nbCells, nbvar);
  Q     = DataArray("Q", nbCells, nbvar);
  
  total_mem_size += nbCells*nbvar * sizeof(real_t) * 3;// 1+1+1 for U+U2+Q
  
  if (params.implementationVersion == 0) {
      
    Fluxes_x = DataArray("Fluxes_x", nbCells, nbvar);
    Fluxes_y = DataArray("Fluxes_y", nbCells, nbvar);

    if (m_dim == 3)
      Fluxes_z = DataArray("Fluxes_z", nbCells, nbvar);

    if (m_dim == 2)
      total_mem_size += nbCells*nbvar * sizeof(real_t) * 2;// 1+1 for Fluxes_x+Fluxes_y
    else
      total_mem_size += nbCells*nbvar * sizeof(real_t) * 3;// 1+1+1 for Fluxes_x+Fluxes_y+Fluxes_z
    
  } else if (params.implementationVersion == 1) {
      
    Slopes_x = DataArray("Slope_x", nbCells, nbvar);
    Slopes_y = DataArray("Slope_y", nbCells, nbvar);

    if (m_dim == 3)
      Slopes_z = DataArray("Slope_z", nbCells, nbvar);

    // direction splitting (only need one flux array)
    Fluxes_x = DataArray("Fluxes_x", nbCells, nbvar);
    Fluxes_y = Fluxes_x;
    if (m_dim == 3)
      Fluxes_z = Fluxes_x;

    if (m_dim==2)
      total_mem_size += nbCells*nbvar * sizeof(real_t) * 3;// 1+1+1 for Slopes_x+Slopes_y+Fluxes_x
    else
      total_mem_size += nbCells*nbvar * sizeof(real_t) * 4;// 1+1+1 for Slopes_x+Slopes_y+Slopes_z+Fluxes_x
    
  } 
  
  // if (m_gravity_enabled) {
  //   gravity = DataArray("gravity field",nbCells,m_dim);
  //   total_mem_size += isize*jsize*2; // TODO
  // }      
 
  // perform init condition
  init(U);
  
  // initialize boundaries
  //make_boundaries(U);

  // copy U into U2
  Kokkos::deep_copy(U2,U);
  
  // compute initialize time step
  compute_dt();

  int myRank=0;
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI

  if (myRank==0) {
    std::cout << "##########################" << "\n";
    std::cout << "Solver is " << m_solver_name << "\n";
    std::cout << "Problem (init condition) is " << m_problem_name << "\n";
    std::cout << "##########################" << "\n";
    
    // print parameters on screen
    params.print();
    std::cout << "##########################" << "\n";
    std::cout << "Memory requested : " << (total_mem_size / 1e6) << " MBytes\n"; 
    std::cout << "##########################" << "\n";
  }

} // SolverHydroMuscl::SolverHydroMuscl

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMuscl::~SolverHydroMuscl()
{

} // SolverHydroMuscl::~SolverHydroMuscl

// =======================================================
// =======================================================
// //////////////////////////////////////////////////
// Fill ghost cells according to border condition :
// absorbant, reflexive or periodic
// //////////////////////////////////////////////////
void SolverHydroMuscl::make_boundaries(DataArray Udata)
{

//   bool mhd_enabled = false;

// #ifdef USE_MPI

//   make_boundaries_mpi(Udata, mhd_enabled);

// #else

//   make_boundaries_serial(Udata, mhd_enabled);
  
// #endif // USE_MPI
  
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

  int level_min = params.level_min;
  
  for (int iter=0; iter<level_min; iter++) {
    amr_mesh->adaptGlobalRefine();
  }

  Kokkos::resize(U,amr_mesh->getNumOctants(),params.nbvar);

  int configNumber = configMap.getInteger("riemann2d","config_number",0);
  real_t xt = configMap.getFloat("riemann2d","x",0.8);
  real_t yt = configMap.getFloat("riemann2d","y",0.8);

  HydroState2d U0, U1, U2, U3;
  getRiemannConfig2d(configNumber, U0, U1, U2, U3);
  
  primToCons_2D(U0, params.settings.gamma0);
  primToCons_2D(U1, params.settings.gamma0);
  primToCons_2D(U2, params.settings.gamma0);
  primToCons_2D(U3, params.settings.gamma0);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);
  auto fm = fieldMgr.get_id2index();

  // Kokkos::parallel_for(amr_mesh->getNumOctants(),
  // 		       KOKKOS_LAMBDA(const size_t &i) {
  // 			 U(i,fm[ID]) = 1.0*i;
  // 			 U(i,fm[IU]) = amr_mesh->getNumOctants()-1.0*i;
  // 			 U(i,fm[IV]) = 1.0*i*i;
  // 			 U(i,fm[IE]) = 1.0*sqrt(i);
  // 		       });

  InitFourQuadrantFunctor::apply(*amr_mesh, params, fm, Udata, configNumber,
				 U0, U1, U2, U3,
				 xt, yt);
  
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
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMuscl::compute_dt_local()
{


} // SolverHydroMuscl::compute_dt_local

// =======================================================
// =======================================================
void SolverHydroMuscl::next_iteration_impl()
{
  
  int myRank=0;
  
#ifdef USE_MPI
  myRank = params.myRank;
#endif // USE_MPI
  
  if (m_iteration % m_nlog == 0) {
    if (myRank==0) {
      printf("time step=%7d (dt=% 10.8f t=% 10.8f)\n",m_iteration,m_dt, m_t);
    }
  }
  
  // output
  if (params.enableOutput) {
    if ( should_save_solution() ) {
      
      if (myRank==0) {
	std::cout << "Output results at time t=" << m_t
		  << " step " << m_iteration
		  << " dt=" << m_dt << std::endl;
      }
      
      save_solution();
      
    } // end output
  } // end enable output
  
  // compute new dt
  m_timers[TIMER_DT]->start();
  compute_dt();
  m_timers[TIMER_DT]->stop();
  
  // perform one step integration
  godunov_unsplit(m_dt);
  
} // SolverHydroMuscl::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverHydroMuscl::godunov_unsplit(real_t dt)
{
  
  if ( m_iteration % 2 == 0 ) {
    godunov_unsplit_impl(U , U2, dt);
  } else {
    godunov_unsplit_impl(U2, U , dt);
  }
  
} // SolverHydroMuscl::godunov_unsplit

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

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverHydroMuscl::convertToPrimitives(DataArray Udata)
{

  
} // SolverHydroMuscl::convertToPrimitives

// =======================================================
// =======================================================
void SolverHydroMuscl::save_solution_impl()
{

  m_timers[TIMER_IO]->start();
  // if (m_iteration % 2 == 0)
  //   save_data(U,  Uhost, m_times_saved, m_t);
  // else
  //   save_data(U2, Uhost, m_times_saved, m_t);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);
  auto fm = fieldMgr.get_id2index();

  // number of macroscopic variables,
  // scalar fields : density, velocity, phase field, etc...
  int nbVar = fieldMgr.numScalarField;

  // a map containing ID and name of the variable to write
  str2int_t names2index; // this is initially empty
  build_var_to_write_map(names2index, params, configMap);
  
  // prepare suffix string
  std::ostringstream strsuf;
  strsuf << "iter";
  strsuf.width(7);
  strsuf.fill('0');
  strsuf << m_iteration;
  
  writeVTK(*amr_mesh, strsuf.str(), U, fm, names2index, configMap);
  
  m_timers[TIMER_IO]->stop();
    
} // SolverHydroMuscl::save_solution_impl()

} // namespace muscl

} // namespace euler_pablo
