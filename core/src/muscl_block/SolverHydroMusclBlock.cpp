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
#include "muscl_block/MarkOctantsHydroFunctor.h"

// // compute functor for low Mach flows
// #include "muscl_block/UpdateRSSTHydroFunctor.h"

// Block data related functors
#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyGhostBlockCellData.h"
#include "shared/mpi/GhostCommunicator.h"

#include "muscl_block/MapUserData.h"

#if BITPIT_ENABLE_MPI==1
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
  interface_flags(),
  Slopes_x(), 
  Slopes_y(), 
  Slopes_z()
#ifdef DYABLO_USE_HDF5
  , hdf5_writer(std::make_shared<HDF5_Writer>(amr_mesh, configMap, params))
#endif // DYABLO_USE_HDF5
{

  solver_type = SOLVER_MUSCL_HANCOCK_BLOCK;
  
  // m_nCells = nbOcts; // TODO
  m_nDofsPerCell = 1;

  int nbvar = params.nbvar;
  int nbfields = params.nbfields;
 
  long long int total_mem_size = 0;

  // Initial number of octants
  // User data will be reallocated after AMR mesh initialization
  uint64_t nbOcts = 1;

  /*
   * setup parameters related to block AMR
   */

  ghostWidth = params.ghostWidth;

  bx = configMap.getInteger("amr", "bx", 0);
  by = configMap.getInteger("amr", "by", 0);
  bz = configMap.getInteger("amr", "bz", 1);

  if (bx < 2*ghostWidth) {
    bx = 2*ghostWidth;
    std::cout << "WARNING: bx should be >= 2*ghostWidth. Setting bx=" << bx << std::endl;
  }
  if (by < 2*ghostWidth) {
    by = 2*ghostWidth;
    std::cout << "WARNING: by should be >= 2*ghostWidth. Setting by=" << by << std::endl;
  }
  if (params.dimType == THREE_D and bz < 2*ghostWidth) {
    bz = 2*ghostWidth;
    std::cout << "WARNING: bz should be >= 2*ghostWidth. Setting bz=" << bz << std::endl;
  }

  bx_g = bx+2*ghostWidth;
  by_g = by+2*ghostWidth;
  bz_g = bz+2*ghostWidth;

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

  U     = DataArrayBlock("U", nbCellsPerOct, nbfields, nbOcts);
  Uhost = Kokkos::create_mirror(U);
  U2    = DataArrayBlock("U2",nbCellsPerOct, nbfields, nbOcts);

  total_mem_size += nbCellsPerOct*nbOcts*nbfields * sizeof(real_t) * 2;// 1+1+1 for U+U2


  // block data array with ghost cells
  Ugroup = DataArrayBlock("Ugroup", nbCellsPerOct_g, nbfields, nbOctsPerGroup);
  Qgroup = DataArrayBlock("Qgroup", nbCellsPerOct_g, nbvar, nbOctsPerGroup);

  total_mem_size += nbCellsPerOct_g*nbOctsPerGroup*nbfields * sizeof(real_t); //Ugroup 
  total_mem_size += nbCellsPerOct_g*nbOctsPerGroup*nbvar    * sizeof(real_t); //Qgroup 

  // flags data array for faces on 2:1 borders
  interface_flags = InterfaceFlags(nbOctsPerGroup);
  total_mem_size += nbOctsPerGroup*sizeof(uint16_t);

  // all intermediate data array are sized upon nbOctsPerGroup

  Slopes_x = DataArrayBlock("Slope_x", nbCellsPerOct_g, nbvar, nbOctsPerGroup);
  Slopes_y = DataArrayBlock("Slope_y", nbCellsPerOct_g, nbvar, nbOctsPerGroup);
  
  if (m_dim == 3)
    Slopes_z = DataArrayBlock("Slope_z", nbCellsPerOct_g, nbvar, nbOctsPerGroup);
  
  if (m_dim==2)
    total_mem_size += nbCellsPerOct_g*nbOctsPerGroup*nbvar * sizeof(real_t) * 2;// 1+1 for Slopes_x+Slopes_y
  else
    total_mem_size += nbCellsPerOct_g*nbOctsPerGroup*nbvar * sizeof(real_t) * 3;// 1+1+1 for Slopes_x+Slopes_y+Slopes_z

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

  lmesh = LightOctree(amr_mesh, params.level_min, params.level_max);

  // compute initialize time step
  compute_dt();

  int myRank=0;
#ifdef DYABLO_USE_MPI
  myRank = params.myRank;
#endif // DYABLO_USE_MPI

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

#ifdef DYABLO_USE_HDF5
  //delete hdf5_writer;
#endif // DYABLO_USE_HDF5

} // SolverHydroMusclBlock::~SolverHydroMusclBlock

// =======================================================
// =======================================================
void SolverHydroMusclBlock::resize_solver_data()
{

  Kokkos::resize(U, nbCellsPerOct, params.nbfields, amr_mesh->getNumOctants());
  Kokkos::resize(U2, nbCellsPerOct, params.nbfields, amr_mesh->getNumOctants());
  Kokkos::resize(Uhost, nbCellsPerOct, params.nbfields, amr_mesh->getNumOctants());
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
      
    } else if ( !m_problem_name.compare("blast") ) {
      
      init_blast(this);
      
    }
    else if ( !m_problem_name.compare("sod") ) {
      
      init_sod(this);
      
    } else if ( !m_problem_name.compare("kelvin_helmholtz") ) {
      
      init_kelvin_helmholtz(this);
      
    } else if ( !m_problem_name.compare("gresho_vortex") ) {
      
      init_gresho_vortex(this);
      
    } else if ( !m_problem_name.compare("four_quadrant") ) {
      
      init_four_quadrant(this);
      
    } else if ( !m_problem_name.compare("isentropic_vortex") ) {
      
      init_isentropic_vortex(this);

    } else if ( !m_problem_name.compare("shu_osher") ) {
      
      init_shu_osher(this);
      
    } else if ( !m_problem_name.compare("double_mach_reflection") ) {

      init_double_mach_reflection(this);
      
    } else if ( !m_problem_name.compare("rayleigh_taylor") ) {
      
      init_rayleigh_taylor(this);
      
    }   
    else if ( !m_problem_name.compare("custom") ) {
      // Don't do anything here, let the user setup their own problem
    } 
    else {
      
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

  timers.get("AMR").start();

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

  timers.get("AMR").stop();

} // SolverHydroMusclBlock::do_amr_cycle

// =======================================================
// =======================================================
void SolverHydroMusclBlock::do_load_balancing()
{

  timers.get("AMR").start();
  
  // load balance
  load_balance_userdata();

  timers.get("AMR").stop();

} // SolverHydroMusclBlock::do_load_balancing

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
  ComputeDtHydroFunctor::apply(lmesh, configMap, 
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
  
#ifdef DYABLO_USE_MPI
  myRank = params.myRank;
#endif // DYABLO_USE_MPI
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
  timers.get("dt").start();
  compute_dt();
  timers.get("dt").stop();
  
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

  if ( should_do_load_balancing() ) {

    do_load_balancing();

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

  // we need conservative variables in ghost cell to be up to date
  synchronize_ghost_data(UserDataCommType::UDATA);

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  // copy data_in into data_out (not necessary)
  // data_out = data_in;
  Kokkos::deep_copy(data_out, data_in);
  
  uint32_t nbOcts = amr_mesh->getNumOctants();

  // number of group of octants, rounding to upper value
  uint32_t nbGroup = (nbOcts + nbOctsPerGroup - 1) / nbOctsPerGroup;

  for (uint32_t iGroup = 0; iGroup < nbGroup; ++iGroup) {

    timers.get("block copy").start();

    // copy data_in (current group of octants) to Ugroup (inner cells)
    fill_block_data_inner(data_in, iGroup);

    // update ghost cells of all octant in current group of octants
    fill_block_data_ghost(data_in, iGroup);

    timers.get("block copy").stop();

    // start main computation
    timers.get("godunov").start();

    // now ghost cells in current group are ok
    // convert conservative variable into primitives ones for the given group
    // input is  Ugroup
    // output is Qgroup
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
    MusclBlockGodunovUpdateFunctor::apply(lmesh,
                                          configMap,
                                          params,
                                          fm,
                                          blockSizes,
                                          ghostWidth,
                                          nbOcts,
                                          nbOctsPerGroup,
                                          iGroup,
                                          Ugroup,
                                          U,
                                          Ughost,
                                          data_out,
                                          Qgroup,
                                          interface_flags,
                                          dt);

    timers.get("godunov").stop();

  } // end for iGroup

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

  timers.get("outputs").start();

  if (params.output_vtk_enabled)
    save_solution_vtk();

  if (params.output_hdf5_enabled)
    save_solution_hdf5();

  timers.get("outputs").stop();
    
} // SolverHydroMusclBlock::save_solution_impl()

// =======================================================
// =======================================================
void SolverHydroMusclBlock::print_monitoring_info()
{

  int myRank = 0;
  int nProcs = 1;
  UNUSED(nProcs);

#ifdef DYABLO_USE_MPI
  myRank = params.myRank;
  nProcs = params.nProcs;
#endif // DYABLO_USE_MPI
  
  // only print on master
  if (myRank == 0) {
    timers.print();

    real_t t_tot   = timers.get("total").elapsed(Timers::Timer::Elapsed_mode_t::ELAPSED_CPU);

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

#ifdef DYABLO_USE_HDF5

  // retrieve available / allowed names: fieldManager, and field map (fm)
  auto fm = fieldMgr.get_id2index();

  // a map containing ID and name of the variable to write
  str2int_t names2index; // this is initially empty
  build_var_to_write_map(names2index, params, configMap);

  // prepare output filename
  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
  std::string outputDir = configMap.getString("output", "outputDir", "./");
  
  // prepare suffix string
  std::ostringstream strsuffix;
  strsuffix << "iter";
  strsuffix.width(7);
  strsuffix.fill('0');
  strsuffix << m_iteration;

  // actual writing
  {

    // resize Uhost upon U
    Kokkos::resize(Uhost, nbCellsPerOct, params.nbfields, amr_mesh->getNumOctants());

    // copy device data to host
    Kokkos::deep_copy(Uhost, U);

    hdf5_writer->update_mesh_info();

    // open the new file and write our stuff
    std::string basename = outputPrefix + "_" + strsuffix.str();
    
    hdf5_writer->open(basename, outputDir);
    hdf5_writer->write_header(m_t);

    // write user the fake data (all scalar fields, here only one)
    hdf5_writer->write_quadrant_attribute(Uhost, fm, names2index);

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
      hdf5_writer->write_quadrant_mach_number(Uhost, fm);
    }

    if (write_variables.find("P") != std::string::npos) {
      hdf5_writer->write_quadrant_pressure(Uhost, fm);
    }

    if (write_variables.find("iOct") != std::string::npos)
      hdf5_writer->write_quadrant_id(Uhost);

    // close the file
    hdf5_writer->write_footer();
    hdf5_writer->close();
  }

#else

  if (amr_mesh->getRank() == 0)
    std::cerr << "You need to re-run cmake and enable HDF5 to have HDF5 output available. Also set hdf5_enabled variable to true in the input paramter file for the run.\n";

#endif // DYABLO_USE_HDF5

} // SolverHydroMusclBlock::save_solution_hdf5

// =======================================================
// =======================================================
void SolverHydroMusclBlock::synchronize_ghost_data(UserDataCommType t)
{

  timers.get("AMR: MPI ghosts").start();

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
    GhostCommunicator comm(amr_mesh);
    comm.exchange_ghosts(U, Ughost);
  }
  default:
    break;
  } // end switch
  
#endif // BITPIT_ENABLE_MPI==1

  timers.get("AMR: MPI ghosts").stop();

} // SolverHydroMusclBlock::synchronize_ghost_data

// =======================================================
// =======================================================
void SolverHydroMusclBlock::mark_cells()
{

  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  real_t error_min = configMap.getFloat("amr", "error_min", 0.2);
  real_t error_max = configMap.getFloat("amr", "error_max", 0.8);

  // TEST HERE !
  DataArrayBlock Udata = U2;

  // apply refinement criterion by parts
  
  uint32_t nbOcts = amr_mesh->getNumOctants();

  // number of group of octants, rounding to upper value
  uint32_t nbGroup = (nbOcts + nbOctsPerGroup - 1) / nbOctsPerGroup;

  MarkOctantsHydroFunctor::markers_t markers(nbOcts);

  for (uint32_t iGroup = 0; iGroup < nbGroup; ++iGroup) {

    timers.get("AMR: block copy").start();

    // copy data_in (current group of octants) to Ugroup (inner cells)
    fill_block_data_inner(Udata, iGroup);

    // update ghost cells of all octant in current group of octants
    fill_block_data_ghost(Udata, iGroup);

    timers.get("AMR: block copy").stop();

    timers.get("AMR: mark cells").start();

    // now ghost cells in current group are ok
    // convert conservative variable into primitives ones for the given group
    // input is  Ugroup
    // output is Qgroup
    convertToPrimitives(iGroup);

    // finaly apply refine criterion : 
    // call device functor to flag for refine/coarsen
    MarkOctantsHydroFunctor::apply(lmesh, configMap, params, fm,
                                   blockSizes, ghostWidth,
                                   nbOcts, nbOctsPerGroup,
                                   Qgroup, iGroup,
                                   error_min, error_max,
                                   markers);

    timers.get("AMR: mark cells").stop();

  } // end for iGroup


  MarkOctantsHydroFunctor::set_markers_pablo(markers, amr_mesh);

} // SolverHydroMusclBlock::mark_cells

// =======================================================
// =======================================================
void SolverHydroMusclBlock::adapt_mesh()
{

  timers.get("AMR: adapt").start();

  // 1. adapt mesh with mapper enabled
  amr_mesh->adapt(true);

  // Verify that adapt() doesn't need another iteration
  assert(amr_mesh->check21Balance());
  assert(!amr_mesh->checkToAdapt());

  // 2. re-compute connectivity
  amr_mesh->updateConnectivity();  
  
  timers.get("AMR: adapt").stop();

} // SolverHydroMusclBlock::adapt_mesh

// =======================================================
// =======================================================
/**
 * input  U2 contains user data before adapt step
 * output U  will be filled with data after remap
 */
void SolverHydroMusclBlock::map_userdata_after_adapt()
{

  timers.get("AMR: map userdata").start();

  LightOctree lmesh_old = lmesh;
  lmesh = LightOctree(amr_mesh, params.level_min, params.level_max);

  MapUserDataFunctor::apply( lmesh_old, lmesh, configMap, blockSizes,
                      U2, Ughost, U );

  // now U contains the most up to date data after mesh adaptation
  // we can resize U2 for the next time-step
  Kokkos::realloc(U2, U.extent(0), U.extent(1), U.extent(2));
  
  timers.get("AMR: map userdata").stop();

} // SolverHydroMusclBlock::map_data_after_adapt

// =======================================================
// =======================================================
void SolverHydroMusclBlock::load_balance_userdata()
{

  timers.get("AMR: load-balance").start();

#if BITPIT_ENABLE_MPI==1

  // retrieve available / allowed names: fieldManager, and field map (fm)
  auto fm = fieldMgr.get_id2index();

  /* (Load)Balance the octree over the processes with communicating the data.
   * Preserve the family compact up to 4 levels over the max deep reached
   * in the octree. */
  {
    uint8_t levels = 3;

    amr_mesh->loadBalance_userdata(levels, U);

    // we probably need to resize arrays, ....
    Kokkos::resize(U2,U.layout());
    Kokkos::realloc(Ughost, Ughost.extent(0), Ughost.extent(1), amr_mesh->getNumGhosts());

    // Update LightOctree after load balancing
    lmesh = LightOctree(amr_mesh, params.level_min, params.level_max);    
  }
#endif // BITPIT_ENABLE_MPI==1
  
  timers.get("AMR: load-balance").stop();

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
                                       data_in, Ugroup, 
                                       iGroup);

} // SolverHydroMusclBlock::fill_block_data_inner

// =======================================================
// =======================================================
void SolverHydroMusclBlock::fill_block_data_ghost(DataArrayBlock data_in,
                                                  uint32_t iGroup)
{
  
  // retrieve available / allowed names: fieldManager, and field map (fm)
  // necessary to access user data
  auto fm = fieldMgr.get_id2index();

  // TODO : use new ghost copy for 2D and 3D 
  //bool use_new_ghost_copy = (params.dimType == THREE_D);
  // if( use_new_ghost_copy )
  // {
    CopyGhostBlockCellDataFunctor::apply(lmesh,
                                        configMap,
                                        params,
                                        fm,
                                        blockSizes,
                                        ghostWidth,
                                        nbOctsPerGroup,
                                        data_in,
                                        Ughost,
                                        Ugroup, 
                                        iGroup,
                                        interface_flags);
  // } else {
  //   // Faces
  //   CopyFaceBlockCellDataFunctor::apply(amr_mesh,
  //                                       configMap,
  //                                       params,
  //                                       fm,
  //                                       blockSizes,
  //                                       ghostWidth,
  //                                       nbOctsPerGroup,
  //                                       data_in,
  //                                       Ughost,
  //                                       Ugroup, 
  //                                       iGroup,
  //                                       interface_flags);

  //   // And corners
  //   CopyCornerBlockCellDataFunctor::apply(amr_mesh,
  //           configMap,
  //           params,
  //           fm,
  //           blockSizes,
  //           ghostWidth,
  //           nbOctsPerGroup,
  //           data_in,
  //           Ughost,
  //           Ugroup,
  //           iGroup,
  //           interface_flags);
  // }
} // SolverHydroMusclBlock::fill_block_data_ghost

} // namespace muscl_block

} // namespace dyablo
