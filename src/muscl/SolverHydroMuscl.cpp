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
#include "muscl/ComputeDtHydroFunctor.h"
#include "muscl/ConvertToPrimitivesHydroFunctor.h"
#include "muscl/ReconstructGradientsHydroFunctor.h"
#include "muscl/ComputeFluxesAndUpdateHydroFunctor.h"
#include "muscl/MarkCellsHydroFunctor.h"

#if BITPIT_ENABLE_MPI==1
#include "muscl/UserDataComm.h"
#include "muscl/UserDataLB.h"
#endif

//#include "shared/mpiBorderUtils.h"

namespace euler_pablo { namespace muscl {

// =======================================================
// ==== CLASS SolverHydroMuscl IMPL ======================
// =======================================================

// =======================================================
// =======================================================
/**
 * \brief SolverHydroMuscl's constructor
 */
SolverHydroMuscl::SolverHydroMuscl(HydroParams& params,
				   ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), Uhost(), U2(), Q(), 
  Ughost(), Qghost(),
  Fluxes_x(), Fluxes_y(), Fluxes_z(),
  Slopes_x(), Slopes_y(), Slopes_z(),
  Slopes_x_ghost(), Slopes_y_ghost(), Slopes_z_ghost()
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

  // init field manager
  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  fieldMgr.setup(params, configMap);

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

  /*
   * this is the initial global refine, to reach level_min / no parallelism
   * so far (every MPI process does that)
   */
  int level_min = params.level_min;
  int level_max = params.level_max;
  
  for (int iter=0; iter<=level_min; iter++) {
    amr_mesh->adaptGlobalRefine();
  }
#if BITPIT_ENABLE_MPI==1
    // (Load)Balance the octree over the MPI processes.
    amr_mesh->loadBalance();
#endif
    //std::cout << "MPI rank=" << amr_mesh->getRank() << " | NB cells =" << amr_mesh->getNumOctants() << "\n";
  

  // genuine initial refinement
  for (int level=level_min; level<level_max; ++level) {

    // mark cells for refinement
    InitBlastRefineFunctor::apply(amr_mesh, configMap, params, level);

    // actually perform refinement
    amr_mesh->adapt();

#if BITPIT_ENABLE_MPI==1
    // (Load)Balance the octree over the MPI processes.
    amr_mesh->loadBalance();
#endif

  } // end for level

  // re-compute mesh connectivity (morton index list, nodes coordinates, ...)
  amr_mesh->updateConnectivity();

  // field manager index array
  auto fm = fieldMgr.get_id2index();

  /*
   * perform user data init
   */
  Kokkos::resize(U,amr_mesh->getNumOctants(),params.nbvar);
  Kokkos::resize(U2,amr_mesh->getNumOctants(),params.nbvar);
  Kokkos::resize(Uhost,amr_mesh->getNumOctants(),params.nbvar);

  Kokkos::resize(Q,amr_mesh->getNumOctants(),params.nbvar);
  
  Kokkos::resize(Slopes_x,amr_mesh->getNumOctants(),params.nbvar);
  Kokkos::resize(Slopes_y,amr_mesh->getNumOctants(),params.nbvar);
  if (params.dimType==THREE_D)
    Kokkos::resize(Slopes_z,amr_mesh->getNumOctants(),params.nbvar);

  InitBlastDataFunctor::apply(amr_mesh, params, configMap, fm, U);

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

  /*
   * this is the initial global refine, to reach level_min / no parallelism
   * so far (every MPI process does that)
   */
  int level_min = params.level_min;
  int level_max = params.level_max;
  
  for (int iter=0; iter<=level_min; iter++) {
    amr_mesh->adaptGlobalRefine();
  }
#if BITPIT_ENABLE_MPI==1
    // (Load)Balance the octree over the MPI processes.
    amr_mesh->loadBalance();
#endif
    //std::cout << "MPI rank=" << amr_mesh->getRank() << " | NB cells =" << amr_mesh->getNumOctants() << "\n";
  
  // load problem specific parameters
  int configNumber = configMap.getInteger("riemann2d","config_number",0);
  real_t xt = configMap.getFloat("riemann2d","x",0.8);
  real_t yt = configMap.getFloat("riemann2d","y",0.8);

  HydroState2d S0, S1, S2, S3;
  getRiemannConfig2d(configNumber, S0, S1, S2, S3);
  
  primToCons_2D(S0, params.settings.gamma0);
  primToCons_2D(S1, params.settings.gamma0);
  primToCons_2D(S2, params.settings.gamma0);
  primToCons_2D(S3, params.settings.gamma0);

  // genuine initial refinement
  for (int level=level_min; level<level_max; ++level) {

    // mark cells for refinement
    InitFourQuadrantRefineFunctor::apply(amr_mesh, params, level, xt, yt);

    // actually perform refinement
    amr_mesh->adapt();

#if BITPIT_ENABLE_MPI==1
    // (Load)Balance the octree over the MPI processes.
    amr_mesh->loadBalance();
#endif

  } // end for level

  // re-compute mesh connectivity (morton index list, nodes coordinates, ...)
  amr_mesh->updateConnectivity();
  
  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  /*
   * perform user data init
   */
  Kokkos::resize(U,amr_mesh->getNumOctants(),params.nbvar);
  Kokkos::resize(U2,amr_mesh->getNumOctants(),params.nbvar);
  Kokkos::resize(Uhost,amr_mesh->getNumOctants(),params.nbvar);

  Kokkos::resize(Q,amr_mesh->getNumOctants(),params.nbvar);
  
  Kokkos::resize(Slopes_x,amr_mesh->getNumOctants(),params.nbvar);
  Kokkos::resize(Slopes_y,amr_mesh->getNumOctants(),params.nbvar);
  if (params.dimType==THREE_D)
    Kokkos::resize(Slopes_z,amr_mesh->getNumOctants(),params.nbvar);

  InitFourQuadrantDataFunctor::apply(amr_mesh, params, fm, U, configNumber,
   				     S0, S1, S2, S3,
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

    if (params.myRank==0)
      printf("Number of octants after init conditions: %ld\n",amr_mesh->getNumOctants());

  } // end regular initialization

} // SolverHydroMuscl::init

// =======================================================
// =======================================================
void SolverHydroMuscl::do_amr_cycle()
{

  /*
   * Following steps:
   *
   * 1. User data comm to update ghost cell values
   * 2. mark cell for refinement / coarsening
   * 3. adapt mesh
   * 4. remap user data to the mesh
   * 5. load balance mesh and user data
   */

  // 1. User data communication / fill ghost data, ghost data
  //    may be needed, e.g. if refine/coarsen condition is a gradient
  synchronize_ghost_data(UserDataCommType::UDATA);
  
  // 2. mark cell for refinement / coarsening + adapt mesh
  mark_cells();

  // 3. adapt mesh + re-compute connectivity
  adapt_mesh();

  // 4. map data to new data array
  map_userdata_after_adapt();

  // 5. load balance
  load_balance_userdata();

} // SolverHydroMuscl::do_amr_cycle

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMuscl::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  // call device functor - compute invDt
  ComputeDtHydroFunctor::apply(amr_mesh, params, fm, U, invDt);

  dt = params.settings.cfl/invDt;

  return dt;

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

  // end of time step, U2 contains next time step data, swap U and U2
  //std::swap(U,U2);

  // deep copy U2 into U
  Kokkos::deep_copy(U,U2);

  // mesh adaptation (perform refine / coarsen)
  if ( should_do_amr_cycle() )
    do_amr_cycle();
    

} // SolverHydroMuscl::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverHydroMuscl::godunov_unsplit(real_t dt)
{
  
  godunov_unsplit_impl(U , U2, dt);
  
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

  // sort of slopes computation adapted to unstructured local mesh
  reconstruct_gradients(data_in);
  
  // communicate borders again, so that slopes in direct neighbors are ok
  // don't care for now about external border - deal with it later
  // m_timers[TIMER_BOUNDARIES]->start();
  // make_boundaries(Slopes_x);
  // make_boundaries(Slopes_y);
  // if (params.dimType==THREE_D)
  //   make_boundaries(Slopes_z);
  // m_timers[TIMER_BOUNDARIES]->stop();

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

  compute_fluxes_and_update(data_in, data_out, dt);

  m_timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMuscl::godunov_unsplit_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverHydroMuscl::convertToPrimitives(DataArray Udata)
{

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  // resize output Q view to the same size as input Udata
  Kokkos::resize(Q, Udata.extent(0), Udata.extent(1));

  // call device functor
  ConvertToPrimitivesHydroFunctor::apply(amr_mesh, params, fm, Udata, Q);
  
} // SolverHydroMuscl::convertToPrimitives

// =======================================================
// =======================================================
void SolverHydroMuscl::reconstruct_gradients(DataArray Udata)
{

  // we need primitive variables in ghost cell to be up to date
  synchronize_ghost_data(UserDataCommType::QDATA);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  // resize slopes views to the same size as input Udata
  Kokkos::resize(Slopes_x, Udata.extent(0), Udata.extent(1));
  Kokkos::resize(Slopes_y, Udata.extent(0), Udata.extent(1));
  if (params.dimType == THREE_D)
    Kokkos::resize(Slopes_z, Udata.extent(0), Udata.extent(1));  

  // call device functor
  ReconstructGradientsHydroFunctor::apply(amr_mesh, params, fm, 
                                          Q, Qghost, Slopes_x, Slopes_y, Slopes_z);
  
} // SolverHydroMuscl::reconstruct_gradients

// =======================================================
// =======================================================
void SolverHydroMuscl::compute_fluxes_and_update(DataArray data_in,
                                                 DataArray data_out,
                                                 real_t dt)
{

  // we need Qdata in ghost update, but this has already been done in 
  // reconstruct_gradients
  // we also need slopes in ghost up to date
  synchronize_ghost_data(UserDataCommType::SLOPES);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  // call device functor
  ComputeFluxesAndUpdateHydroFunctor::apply(amr_mesh, params, fm, 
                                            data_in, data_out, 
                                            Q, Qghost, 
                                            Slopes_x, 
                                            Slopes_y, 
                                            Slopes_z,
                                            Slopes_x_ghost, 
                                            Slopes_y_ghost,
                                            Slopes_z_ghost,
                                            dt);
  
} // SolverHydroMuscl::compute_fluxes_and_update

// =======================================================
// =======================================================
void SolverHydroMuscl::save_solution_impl()
{

  m_timers[TIMER_IO]->start();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  auto fm = fieldMgr.get_id2index();

  // number of macroscopic variables,
  // scalar fields : density, velocity, phase field, etc...
  //int nbVar = fieldMgr.numScalarField;

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

  if (params.debug_output) {
    writeVTK(*amr_mesh, strsuf.str(), Slopes_x, fm, names2index, configMap, "_slope_x");
    writeVTK(*amr_mesh, strsuf.str(), Slopes_y, fm, names2index, configMap, "_slope_y");
  }

  m_timers[TIMER_IO]->stop();
    
} // SolverHydroMuscl::save_solution_impl()

// =======================================================
// =======================================================
void SolverHydroMuscl::synchronize_ghost_data(UserDataCommType t)
{

  // retrieve available / allowed names: fieldManager, and field map (fm)
  auto fm = fieldMgr.get_id2index();

  // retrieve current number of ghost cells
  uint32_t nghosts = amr_mesh->getNumGhosts();

#if BITPIT_ENABLE_MPI==1

  // select which data to exchange

  // 3 operations :
  // 1. resize ghost array
  // 2. create UserDataComm object
  // 3. perform MPI communications
  
  switch(t) {
  case UserDataCommType::UDATA: {
    Kokkos::resize(Ughost, nghosts, U.extent(1));
    UserDataComm data_comm(U, Ughost, fm);
    amr_mesh->communicate(data_comm);
    break;
  }
  case UserDataCommType::QDATA : {
    Kokkos::resize(Qghost, nghosts, Q.extent(1));
    UserDataComm data_comm(Q, Qghost, fm);
    amr_mesh->communicate(data_comm);
    break;
  }
  case UserDataCommType::SLOPES : {
    {
      Kokkos::resize(Slopes_x_ghost, nghosts, Q.extent(1));
      UserDataComm data_comm(Slopes_x, Slopes_x_ghost, fm);
      amr_mesh->communicate(data_comm);
    }
    {
      Kokkos::resize(Slopes_y_ghost, nghosts, Q.extent(1));
      UserDataComm data_comm(Slopes_y, Slopes_y_ghost, fm);
      amr_mesh->communicate(data_comm);
    }
    if (params.dimType==THREE_D) {
      Kokkos::resize(Slopes_z_ghost, nghosts, Q.extent(1));
      UserDataComm data_comm(Slopes_z, Slopes_z_ghost, fm);
      amr_mesh->communicate(data_comm);
    }
    
  } // end case SLOPES

  } // end switch
  
#endif

} // SolverHydroMuscl::synchronize_ghost_data

// =======================================================
// =======================================================
void SolverHydroMuscl::mark_cells()
{

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  real_t eps_refine  = configMap.getFloat("amr", "epsilon_refine", 0.001);
  real_t eps_coarsen = configMap.getFloat("amr", "epsilon_coarsen", 0.002);

  DataArray Udata = U2;

  // Note: Ughost is up to date, update at the beginning of do_amr_cycle

  // call device functor to flag for refine/coarsen
  MarkCellsHydroFunctor::apply(amr_mesh, params, fm, Udata, Ughost,
                               eps_refine, eps_coarsen);
  
} // SolverHydroMuscl::mark_cells

// =======================================================
// =======================================================
void SolverHydroMuscl::adapt_mesh()
{

  // 1. adapt mesh
  amr_mesh->adapt(true);

  // 2. re-compute connectivity
  amr_mesh->updateConnectivity();
  
} // SolverHydroMuscl::adapt_mesh

// =======================================================
// =======================================================
void SolverHydroMuscl::map_userdata_after_adapt()
{

  // TODO : make is mapper and isghost Kokkos::View's so that
  // one can make the rest of this routine parallel
  std::vector<uint32_t> mapper;
  std::vector<bool> isghost;
  
  // 
  int nbVars = params.nbvar;

    // field manager index array
  auto fm = fieldMgr.get_id2index();

  // at this stage, the numerical scheme has been computed
  // U contains data at t_n
  // U2 contains data at t_{n+1}
  //
  // so let's just resize U, and remap U2 to U after the mesh adaptation

  //amr_mesh->adapt(true);
  uint32_t nocts = amr_mesh->getNumOctants();
  Kokkos::resize(U, nocts, nbVars);
  
  // reset U
  Kokkos::parallel_for(nocts, KOKKOS_LAMBDA(const size_t i) {
      for (int ivar=0; ivar<nbVars; ++ivar)
        U(i,fm[ivar])=0.0;
    });
  
  /*
   * Assign to the new octant the average of the old children
   *  if it is new after a coarsening;
   * while assign to the new octant the data of the old father
   *  if it is new after a refinement.
   */
  // TODO : make this loop a parallel_for ?
  for (uint32_t i=0; i<nocts; i++) {
    
    amr_mesh->getMapping(i, mapper, isghost);
    
    if (amr_mesh->getIsNewC(i)) {

      for (int j=0; j<m_nbChildren; ++j) {

	if (isghost[j]) {
	  
          for (int ivar=0; ivar<nbVars; ++ivar)
	    U(i,fm[ivar]) += Ughost(mapper[j],fm[ivar])/m_nbChildren;

	} else {

	  for (int ivar=0; ivar<nbVars; ++ivar)
	    U(i,fm[ivar]) += U2(mapper[j],fm[ivar])/m_nbChildren;
	}
      }

    } else {

      for (int ivar=0; ivar<nbVars; ++ivar)
	U(i,fm[ivar]) = U2(mapper[0],fm[ivar]);

    }
  }

  // now U contains the most up to date data after mesh adaptation
  // we can resize U2 for the next time-step
  Kokkos::resize(U2, U.extent(0), U.extent(1));

  // re-assign U_new to U
  //U = U_new;
  
} // SolverHydroMuscl::map_data_after_adapt

// =======================================================
// =======================================================
void SolverHydroMuscl::load_balance_userdata()
{
  
#if BITPIT_ENABLE_MPI==1
  /* (Load)Balance the octree over the processes with communicating the data.
   * Preserve the family compact up to 4 levels over the max deep reached
   * in the octree. */
  {
    uint8_t levels = 4;
    UserDataLB data_lb(U, Ughost);
    amr_mesh->loadBalance(data_lb, levels);
  }
#endif // BITPIT_ENABLE_MPI==1
  
} // SolverHydroMuscl::load_balance_user_data


} // namespace muscl

} // namespace euler_pablo
