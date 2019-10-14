/**
 * \file SolverHydroMusclBlock.cpp
 *
 * \author Pierre Kestener
 */
#include <string>
#include <cstdio>
#include <cstdbool>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "shared/HydroParams.h"

#include "shared/SimpleVTKIO.h"

#include "muscl_block/SolverHydroMusclBlock.h"

// Init conditions functors
#include "muscl_block/init/HydroInitFunctors.h"

// Compute functors
#include "muscl_block/ComputeDtHydroFunctor.h"
#include "muscl_block/ConvertToPrimitivesHydroFunctor.h"
#include "muscl_block/MusclBlockGodunovUpdateFunctor.h"
#include "muscl_block/MusclBlockSharedGodunovUpdateFunctor.h"
#include "muscl_block/MarkOctantsHydroFunctor.h"

// // compute functor for low Mach flows
// #include "muscl_block/UpdateRSSTHydroFunctor.h"

// Block data related functors
#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyFaceBlockCellData.h"

#if BITPIT_ENABLE_MPI==1
#include "muscl_block/UserDataComm.h"
#include "muscl_block/UserDataLB.h"
#endif

//#include "shared/mpiBorderUtils.h"

namespace dyablo { namespace muscl_block {

// =======================================================
// ==== CLASS SolverHydroMusclBlock IMPL =================
// =======================================================

// =======================================================
// =======================================================
/**
 * \brief SolverHydroMusclBlock's constructor
 */
SolverHydroMusclBlock::SolverHydroMusclBlock(HydroParams& params,
                                             ConfigMap& configMap) :
  SolverBase(params, configMap),
  U(), Uhost(), U2(), Ughost(), 
  Ugroup(), 
  Qgroup(),
  Slopes_x(), 
  Slopes_y(), 
  Slopes_z()
#ifdef USE_HDF5
  , hdf5_writer(std::make_shared<HDF5_Writer>(amr_mesh, configMap, params))
#endif // USE_HDF5
{

  solver_type = SOLVER_MUSCL_HANCOCK_BLOCK;
  
  // m_nCells = nbOcts; // TODO
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

  // minimal number of cells only used for initial memory allocation
  // afterwards memory resizing will append
  uint64_t nbOcts = 1<<params.level_min;
  
  nbOcts = params.dimType == TWO_D ? nbOcts * nbOcts :  nbOcts * nbOcts * nbOcts;

  /*
   * setup parameters related to block AMR
   */

  ghostWidth = configMap.getInteger("amr", "ghostwidth", 2);

  bx = configMap.getInteger("amr", "bx", 0);
  by = configMap.getInteger("amr", "by", 0);
  bz = configMap.getInteger("amr", "bz", 1);

  bx_g = bx+2*ghostWidth;
  by_g = bx+2*ghostWidth;
  bz_g = bx+2*ghostWidth;

  blockSizes[IX] = bx;
  blockSizes[IY] = by;
  blockSizes[IZ] = bz;

  blockSizes_g[IX] = bx_g;
  blockSizes_g[IY] = by_g;
  blockSizes_g[IZ] = bz_g;

  nbCellsPerOct   = params.dimType == TWO_D ? bx  *by   : bx  *by  *bz;
  nbCellsPerOct_g = params.dimType == TWO_D ? bx_g*by_g : bx_g*by_g*bz_g;

  nbOctsPerGroup = configMap.getInteger("amr", "nbOctsPerGroup", 32);

  /*
   * main data array memory allocation
   */

  U     = DataArrayBlock("U", nbCellsPerOct, nbvar, nbOcts);
  Uhost = Kokkos::create_mirror(U);
  U2    = DataArrayBlock("U2",nbCellsPerOct, nbvar, nbOcts);
  
  total_mem_size += nbCellsPerOct*nbOcts*nbvar * sizeof(real_t) * 2;// 1+1+1 for U+U2


  // block data array with ghost cells
  Ugroup = DataArrayBlock("Ugroup", nbCellsPerOct_g, nbvar, nbOctsPerGroup);
  Qgroup = DataArrayBlock("Qgroup", nbCellsPerOct_g, nbvar, nbOctsPerGroup);

  total_mem_size += nbCellsPerOct_g*nbOctsPerGroup*nbvar * sizeof(real_t) * 2 ;// 1+1 for Ugroup and Qgroup


  // all intermediate data array are sized upon nbOctsPerGroup

  Slopes_x = DataArrayBlock("Slope_x", nbCellsPerOct_g, nbvar, nbOctsPerGroup);
  Slopes_y = DataArrayBlock("Slope_y", nbCellsPerOct_g, nbvar, nbOctsPerGroup);
  
  if (m_dim == 3)
    Slopes_z = DataArrayBlock("Slope_z", nbCellsPerOct_g, nbvar, nbOctsPerGroup);
  
  if (m_dim==2)
    total_mem_size += nbCellsPerOct_g*nbOctsPerGroup*nbvar * sizeof(real_t) * 2;// 1+1 for Slopes_x+Slopes_y
  else
    total_mem_size += nbCellsPerOct_g*nbOctsPerGroup*nbvar * sizeof(real_t) * 3;// 1+1+1 for Slopes_x+Slopes_y+Slopes_z
  
  
  // if (m_gravity_enabled) {
  //   gravity = DataArrayBlock("gravity field",nbOcts,m_dim);
  //   total_mem_size += isize*jsize*2; // TODO
  // }

  if (params.rsst_enabled) {
    Fluxes = DataArrayBlock("Fluxes", nbCellsPerOct, nbvar, nbOctsPerGroup);
    total_mem_size += nbCellsPerOct * nbOctsPerGroup * nbvar * sizeof(real_t); //
  }

  // init field manager
  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  fieldMgr.setup(params, configMap);

  // perform init condition
  init(U);
  
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

} // SolverHydroMusclBlock::SolverHydroMusclBlock

// =======================================================
// =======================================================
/**
 *
 */
SolverHydroMusclBlock::~SolverHydroMusclBlock()
{

#ifdef USE_HDF5
  //delete hdf5_writer;
#endif // USE_HDF5

} // SolverHydroMusclBlock::~SolverHydroMusclBlock

// =======================================================
// =======================================================
void SolverHydroMusclBlock::resize_solver_data()
{

  Kokkos::resize(U, nbCellsPerOct, params.nbvar, amr_mesh->getNumOctants());
  Kokkos::resize(U2, nbCellsPerOct, params.nbvar, amr_mesh->getNumOctants());
  Kokkos::resize(Uhost, nbCellsPerOct, params.nbvar, amr_mesh->getNumOctants());

  // Remember that all other array are fixed sized - nbOctsPerGroup

} // SolverHydroMusclBlock::resize_solver_data

// =======================================================
// =======================================================
void SolverHydroMusclBlock::init_restart(DataArrayBlock Udata)
{

  // TODO
  
} // SolverHydroMusclBlock::init_restart

// =======================================================
// =======================================================
void SolverHydroMusclBlock::init(DataArrayBlock Udata)
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
      
      init_implode(this);
      
    } else if ( !m_problem_name.compare("sod") ) {
      
      //init_sod(this);
      
    } else if ( !m_problem_name.compare("blast") ) {
      
      init_blast(this);
      
    } else if ( !m_problem_name.compare("kelvin_helmholtz") ) {
      
      //init_kelvin_helmholtz(this);
      
    } else if ( !m_problem_name.compare("gresho_vortex") ) {
      
      init_gresho_vortex(this);
      
    } else if ( !m_problem_name.compare("four_quadrant") ) {
      
      //init_four_quadrant(this);
      
    } else if ( !m_problem_name.compare("isentropic_vortex") ) {
      
      //init_isentropic_vortex(this);
      
    } else if ( !m_problem_name.compare("rayleigh_taylor") ) {
      
      //init_rayleigh_taylor(this);
      
    } else {
      
      std::cout << "Problem : " << m_problem_name
		<< " is not recognized / implemented."
		<< std::endl;
      std::cout <<  "Use default - implode" << std::endl;
      init_implode(this);
      
    }

    // initialize U2
    Kokkos::deep_copy(U2,U);

  } // end regular initialization

} // SolverHydroMusclBlock::init

// =======================================================
// =======================================================
void SolverHydroMusclBlock::do_amr_cycle()
{

  m_timers[TIMER_AMR_CYCLE]->start();

  /*
   * Following steps:
   *
   * 1. MPI synchronize user data to update ghost cell values
   * 2. mark cell for refinement / coarsening (requiring up to date ghost)
   * 3. adapt mesh
   * 4. remap user data to the new mesh
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

  m_timers[TIMER_AMR_CYCLE]->stop();

} // SolverHydroMusclBlock::do_amr_cycle

// =======================================================
// =======================================================
/**
 * Compute time step satisfying CFL condition.
 *
 * \return dt time step
 */
double SolverHydroMusclBlock::compute_dt_local()
{

  real_t dt;
  real_t invDt = ZERO_F;

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  // call device functor - compute invDt
  ComputeDtHydroFunctor::apply(amr_mesh, configMap, 
                               params, fm,
                               blockSizes, U, invDt);

  dt = params.settings.cfl/invDt;

  return dt;

} // SolverHydroMusclBlock::compute_dt_local

// =======================================================
// =======================================================
void SolverHydroMusclBlock::next_iteration_impl()
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

  // mesh adaptation (perform refine / coarsen)
  // at the end: most current data will be stored in U
  // amr cycle is automatically disabled when level_min == level_max
  if ( should_do_amr_cycle() ) {

    // before do_amr_cycle, up to date data are in U2
    // after  do_amr_cycle, up to date data are in U
    do_amr_cycle();

  } else {
    
    // just deep copy U2 into U 
    Kokkos::deep_copy(U,U2);

  }

  // TODO
  // if ( should_write_restart_file() )
  //   write_restart_file();

} // SolverHydroMusclBlock::next_iteration_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Wrapper to the actual computation routine
// ///////////////////////////////////////////
void SolverHydroMusclBlock::godunov_unsplit(real_t dt)
{
  
  godunov_unsplit_impl(U , U2, dt);

} // SolverHydroMusclBlock::godunov_unsplit

// =======================================================
// =======================================================
// ///////////////////////////////////////////
// Actual computation of Godunov scheme
// ///////////////////////////////////////////
void SolverHydroMusclBlock::godunov_unsplit_impl(DataArrayBlock data_in, 
                                                 DataArrayBlock data_out, 
                                                 real_t dt)
{

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  // start main computation
  m_timers[TIMER_NUM_SCHEME]->start();

  uint32_t nbOcts = amr_mesh->getNumOctants();

  // number of group of octants, rounding to upper value
  uint32_t nbGroup = (nbOcts + nbOctsPerGroup - 1) / nbOctsPerGroup;

  for (uint32_t iGroup = 0; iGroup < nbGroup; ++iGroup) {

    m_timers[TIMER_BLOCK_COPY]->start();

    // copy data_in (current group of octants) to Ugroup (inner cells)
    fill_block_data_inner(data_in, iGroup);

    // update ghost cells of all octant in current group of octants
    fill_block_data_ghost(data_in, iGroup);

    m_timers[TIMER_BLOCK_COPY]->stop();

    // now ghost cells in current group are ok
    // convert conservative variable into primitives ones for the entire domain
    convertToPrimitives(iGroup);

    // perform time integration :

    /*
     * algorithmic variant using shared memory, but no extra
     * heap memory
     */
    // MusclBlockSharedGodunovUpdateFunctor::apply(amr_mesh,
    //                                             configMap,
    //                                             params,
    //                                             fm,
    //                                             blockSizes,
    //                                             ghostWidth,
    //                                             nbOcts,
    //                                             nbOctsPerGroup,
    //                                             iGroup,
    //                                             Ugroup,
    //                                             data_out,
    //                                             Qgroup,
    //                                             dt);

    /*
     * algorithmic variant not using shared memory, so extra
     * heap memory is required (array SlopesX, ... are regular
     * Kokkos::View arrays sized upon the group of octant)
     */
    MusclBlockGodunovUpdateFunctor::apply(amr_mesh,
                                          configMap,
                                          params,
                                          fm,
                                          blockSizes,
                                          ghostWidth,
                                          nbOcts,
                                          nbOctsPerGroup,
                                          iGroup,
                                          Ugroup,
                                          data_out,
                                          Qgroup,
                                          dt);

  } // end for iGroup

  m_timers[TIMER_NUM_SCHEME]->stop();
  
} // SolverHydroMusclBlock::godunov_unsplit_impl

// =======================================================
// =======================================================
// ///////////////////////////////////////////////////////////////////
// Convert conservative variables array U into primitive var array Q
// ///////////////////////////////////////////////////////////////////
void SolverHydroMusclBlock::convertToPrimitives(uint32_t iGroup)
{

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  uint32_t nbOcts = amr_mesh->getNumOctants();

  // call device functor
  ConvertToPrimitivesHydroFunctor::apply(configMap,
                                         params, 
                                         fm,
                                         blockSizes,
                                         ghostWidth,
                                         nbOcts,
                                         nbOctsPerGroup,
                                         iGroup,
                                         Ugroup, 
                                         Qgroup);

  
} // SolverHydroMusclBlock::convertToPrimitives

// =======================================================
// =======================================================
void SolverHydroMusclBlock::save_solution_impl()
{

  m_timers[TIMER_IO]->start();

  if (params.output_vtk_enabled)
    save_solution_vtk();

  if (params.output_hdf5_enabled)
    save_solution_hdf5();

  m_timers[TIMER_IO]->stop();
    
} // SolverHydroMusclBlock::save_solution_impl()

// =======================================================
// =======================================================
void SolverHydroMusclBlock::print_monitoring_info()
{

  real_t t_tot   = m_timers[TIMER_TOTAL]->elapsed();
  real_t t_comp  = m_timers[TIMER_NUM_SCHEME]->elapsed();
  real_t t_dt    = m_timers[TIMER_DT]->elapsed();
  real_t t_bound = m_timers[TIMER_BOUNDARIES]->elapsed();
  real_t t_io    = m_timers[TIMER_IO]->elapsed();
  real_t t_amr   = m_timers[TIMER_AMR_CYCLE]->elapsed();

  real_t t_amr_sync_ghost   = m_timers[TIMER_AMR_CYCLE_SYNC_GHOST]->elapsed();
  real_t t_amr_mark_cells   = m_timers[TIMER_AMR_CYCLE_MARK_CELLS]->elapsed();
  real_t t_amr_adapt_mesh   = m_timers[TIMER_AMR_CYCLE_ADAPT_MESH]->elapsed();
  real_t t_amr_map_userdata = m_timers[TIMER_AMR_CYCLE_MAP_USERDATA]->elapsed();
  real_t t_amr_load_balance = m_timers[TIMER_AMR_CYCLE_LOAD_BALANCE]->elapsed();

  real_t t_block_copy = m_timers[TIMER_BLOCK_COPY]->elapsed();

  int myRank = 0;
  int nProcs = 1;
  UNUSED(nProcs);

#ifdef USE_MPI
  myRank = params.myRank;
  nProcs = params.nProcs;
#endif // USE_MPI
  
  // only print on master
  if (myRank == 0) {

    printf("total       time : %5.3f secondes\n", t_tot);
    printf("godunov     time : %5.3f secondes %5.2f%%\n", t_comp,
           100 * t_comp / t_tot);
    printf("compute dt  time : %5.3f secondes %5.2f%%\n", t_dt,
           100 * t_dt / t_tot);
    printf("boundaries  time : %5.3f secondes %5.2f%%\n", t_bound,
           100 * t_bound / t_tot);
    printf("io          time : %5.3f secondes %5.2f%%\n", t_io,
           100 * t_io / t_tot);

    printf("block copy  time : %5.3f secondes %5.2f%%\n", t_block_copy,
           100 * t_block_copy / t_tot);

    printf("amr cycle   time : %5.3f secondes %5.2f%%\n", t_amr,
           100 * t_amr / t_tot);

    printf("amr cycle sync ghost    : %5.3f secondes %5.2f%%\n",
           t_amr_sync_ghost, 100 * t_amr_sync_ghost / t_tot);
    printf("amr cycle mark cells    : %5.3f secondes %5.2f%%\n",
           t_amr_mark_cells, 100 * t_amr_mark_cells / t_tot);
    printf("amr cycle adapt mesh    : %5.3f secondes %5.2f%%\n",
           t_amr_adapt_mesh, 100 * t_amr_adapt_mesh / t_tot);
    printf("amr cycle map user data : %5.3f secondes %5.2f%%\n",
           t_amr_map_userdata, 100 * t_amr_map_userdata / t_tot);
    printf("amr cycle load balance  : %5.3f secondes %5.2f%%\n",
           t_amr_load_balance, 100 * t_amr_load_balance / t_tot);

    printf("Perf             : %5.3f number of Mcell-updates/s\n",
           1.0 * m_total_num_cell_updates * (bx*by*bz) / t_tot * 1e-6);

    printf("Total number of cell-updates : %ld\n",
           m_total_num_cell_updates * (bx*by*bz));

  } // end myRank==0

} // SolverHydroMusclBlock::print_monitoring_info

// =======================================================
// =======================================================
void SolverHydroMusclBlock::save_solution_vtk() 
{

  std::cerr << "writeVTK for block AMR is not implemented - TODO / REALLY USEFUL ?\n";

} // SolverHydroMusclBlock::save_solution_vtk

// =======================================================
// =======================================================
void SolverHydroMusclBlock::save_solution_hdf5() 
{

#ifdef USE_HDF5

  // retrieve available / allowed names: fieldManager, and field map (fm)
  auto fm = fieldMgr.get_id2index();

  // a map containing ID and name of the variable to write
  str2int_t names2index; // this is initially empty
  build_var_to_write_map(names2index, params, configMap);

  // prepare output filename
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  // prepare suffix string
  std::ostringstream strsuffix;
  strsuffix << "iter";
  strsuffix.width(7);
  strsuffix.fill('0');
  strsuffix << m_iteration;
  
  // actual writing
  {

    hdf5_writer->update_mesh_info();

    // open the new file and write our stuff
    hdf5_writer->open(outputPrefix + "_" + strsuffix.str());
    hdf5_writer->write_header(m_t);

    // write user the fake data (all scalar fields, here only one)
    hdf5_writer->write_quadrant_attribute(U, fm, names2index);

    // check if we want to write velocity or rhoV vector fields
    std::string write_variables = configMap.getString("output", "write_variables", "");
    // if (write_variables.find("velocity") != std::string::npos) {
    //   hdf5_writer->write_quadrant_velocity(U, fm, false);
    // } else if (write_variables.find("rhoV") != std::string::npos) {
    //   hdf5_writer->write_quadrant_velocity(U, fm, true);
    // } 
    
    if (write_variables.find("Mach") != std::string::npos) {
      // mach number will be recomputed from conservative variables
      // we could have used primitive variables, but since here Q
      // may not have the same size, Q may need to be resized
      // and recomputed anyway.
      hdf5_writer->write_quadrant_mach_number(U, fm);
    }

    // close the file
    hdf5_writer->write_footer();
    hdf5_writer->close();
  }

#else

  if (amr_mesh->getRank() == 0)
    std::cerr << "You need to re-run cmake and enable HDF5 to have HDF5 output available. Also set hdf5_enabled variable to true in the input paramter file for the run.\n";

#endif // USE_HDF5

} // SolverHydroMusclBlock::save_solution_hdf5

// =======================================================
// =======================================================
void SolverHydroMusclBlock::synchronize_ghost_data(UserDataCommType t)
{

  m_timers[TIMER_AMR_CYCLE_SYNC_GHOST]->start();

#if BITPIT_ENABLE_MPI==1

  // retrieve available / allowed names: fieldManager, and field map (fm)
  auto fm = fieldMgr.get_id2index();

  // retrieve current number of ghost cells
  uint32_t nghosts = amr_mesh->getNumGhosts();

  // select which data to exchange

  // 3 operations :
  // 1. resize ghost array
  // 2. create UserDataComm object
  // 3. perform MPI communications
  
  switch(t) {
  case UserDataCommType::UDATA: {
    Kokkos::resize(Ughost, U.extent(0), U.extent(1), nghosts);
    UserDataComm data_comm(U, Ughost, fm);
    amr_mesh->communicate(data_comm);
    break;
  }
  default:
    break;
  } // end switch
  
#endif // BITPIT_ENABLE_MPI==1

  m_timers[TIMER_AMR_CYCLE_SYNC_GHOST]->stop();

} // SolverHydroMusclBlock::synchronize_ghost_data

// =======================================================
// =======================================================
void SolverHydroMusclBlock::mark_cells()
{

  m_timers[TIMER_AMR_CYCLE_MARK_CELLS]->start();

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  //auto fm = fieldMgr.get_id2index();

  //real_t eps_refine  = configMap.getFloat("amr", "epsilon_refine", 0.001);
  //real_t eps_coarsen = configMap.getFloat("amr", "epsilon_coarsen", 0.002);

  //DataArrayBlock Udata = U2;

  // Note: Ughost is up to date, update at the beginning of do_amr_cycle

  // TODO
  // call device functor to flag for refine/coarsen
  // MarkCellsHydroFunctor::apply(amr_mesh, params, fm, Udata, Ughost,
  //                              eps_refine, eps_coarsen);

  m_timers[TIMER_AMR_CYCLE_MARK_CELLS]->stop();

} // SolverHydroMusclBlock::mark_cells

// =======================================================
// =======================================================
void SolverHydroMusclBlock::adapt_mesh()
{

  m_timers[TIMER_AMR_CYCLE_ADAPT_MESH]->start();

  // 1. adapt mesh with mapper enabled
  amr_mesh->adapt(true);

  // 2. re-compute connectivity
  amr_mesh->updateConnectivity();
  
  m_timers[TIMER_AMR_CYCLE_ADAPT_MESH]->stop();

} // SolverHydroMusclBlock::adapt_mesh

// =======================================================
// =======================================================
/**
 * input  U2 contains user data before adapt step
 * output U  will be filled with data after remap
 */
void SolverHydroMusclBlock::map_userdata_after_adapt()
{

  m_timers[TIMER_AMR_CYCLE_MAP_USERDATA]->start();

  // TODO : refactor
  // turn mapper and isghost into Kokkos::View's (instead of
  // std::vector) so that
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
  Kokkos::resize(U, nbCellsPerOct, nbVars, nocts);
  
  // reset U
  Kokkos::parallel_for("dyablo::muscl_block::SolverHydroMusclBlock reset U",nocts, KOKKOS_LAMBDA(const size_t iOct) {
      for (int ivar=0; ivar<nbVars; ++ivar)
        for (uint32_t index=0; index<nbCellsPerOct; ++index)
          U(index,fm[ivar],iOct)=0.0;
    });
  
  /*
   * Assign to the new octant the average of the old children
   *  if it is new after a coarsening;
   * while assign to the new octant the data of the old father
   *  if it is new after a refinement.
   *
   * Important, here we assume mapper containts octant id in the Morton order.
   * We actually need to know how the old cells are located
   * relatively one to another to perform correct remapping.
   */
  // TODO : make this loop a parallel_for ?
  for (uint32_t iOct=0; iOct<nocts; iOct++) {

    // fill mapper and isghost for current octant
    amr_mesh->getMapping(iOct, mapper, isghost);

    // test is current cell is new upon a coarsening operation
    if ( amr_mesh->getIsNewC(iOct) ) {

      for (int iOctChild=0; iOctChild<m_nbChildren; ++iOctChild) {

	if (isghost[iOctChild]) {
	  
          // for (int ivar=0; ivar<nbVars; ++ivar)
	  //   U(i,fm[ivar]) += Ughost(mapper[j],ivar)/m_nbChildren;

        } else {

          // for (int ivar = 0; ivar < nbVars; ++ivar)
          //   U(i, fm[ivar]) += U2(mapper[j], fm[ivar]) / m_nbChildren;

        }

      }

    } else {
      
      // current cell is just an old cell or new upon a refinement,
      // so we just copy data
      
      // for (int ivar = 0; ivar < nbVars; ++ivar)
      //   U(i, fm[ivar]) = U2(mapper[0], fm[ivar]);

    }
  }

  // now U contains the most up to date data after mesh adaptation
  // we can resize U2 for the next time-step
  Kokkos::resize(U2, U.extent(0), U.extent(1), U.extent(2));
  
  m_timers[TIMER_AMR_CYCLE_MAP_USERDATA]->stop();

} // SolverHydroMusclBlock::map_data_after_adapt

// =======================================================
// =======================================================
void SolverHydroMusclBlock::load_balance_userdata()
{

  m_timers[TIMER_AMR_CYCLE_LOAD_BALANCE]->start();

#if BITPIT_ENABLE_MPI==1

  // retrieve available / allowed names: fieldManager, and field map (fm)
  auto fm = fieldMgr.get_id2index();

  /* (Load)Balance the octree over the processes with communicating the data.
   * Preserve the family compact up to 4 levels over the max deep reached
   * in the octree. */
  {
    uint8_t levels = 4;

    UserDataLB data_lb(U, Ughost, fm);
    amr_mesh->loadBalance(data_lb, levels);

    // we probably need to resize U2, ....
    Kokkos::resize(U2,U.extent(0),U.extent(1),U.extent(2));

  }
#endif // BITPIT_ENABLE_MPI==1
  
  m_timers[TIMER_AMR_CYCLE_LOAD_BALANCE]->stop();

} // SolverHydroMusclBlock::load_balance_user_data

// =======================================================
// =======================================================
void SolverHydroMusclBlock::fill_block_data_inner(DataArrayBlock data_in,
                                                  uint32_t iGroup)
{

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  uint32_t nbOcts = amr_mesh->getNumOctants();
  
  CopyInnerBlockCellDataFunctor::apply(configMap, params, fm,
                                       blockSizes,
                                       ghostWidth,
                                       nbOcts,
                                       nbOctsPerGroup,
                                       data_in, Ugroup, iGroup);

} // SolverHydroMusclBlock::fill_block_data_inner

// =======================================================
// =======================================================
void SolverHydroMusclBlock::fill_block_data_ghost(DataArrayBlock data_in,
                                                  uint32_t iGroup)
{
  
  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  CopyFaceBlockCellDataFunctor::apply(amr_mesh,
                                      configMap,
                                      params,
                                      fm,
                                      blockSizes,
                                      ghostWidth,
                                      nbOctsPerGroup,
                                      data_in,
                                      Ughost,
                                      Ugroup, 
                                      iGroup);

} // SolverHydroMusclBlock::fill_block_data_ghost

} // namespace muscl_block

} // namespace dyablo
