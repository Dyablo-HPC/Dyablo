#include "HydroParams.h"

#include <cstdlib> // for exit
#include <cstdio>  // for fprintf
#include <cstring> // for strcmp
#include <iostream>

#include "config/inih/ini.h" // our INI file reader
#include "utils/mpi/GlobalMpiSession.h"

// =======================================================
// =======================================================
/*
 * Hydro Parameters (read parameter file)
 */
void HydroParams::setup(ConfigMap &configMap)
{

  /* initialize RUN parameters */
  nStepmax = configMap.getInteger("run","nstepmax",1000);
  tEnd     = configMap.getFloat  ("run","tend",0.0);
  nOutput  = configMap.getInteger("run","noutput",100);
  if (nOutput == 0)
    enableOutput = false;

  nlog     = configMap.getInteger("run","nlog",10);
  
  std::string solver_name = configMap.getString("run", "solver_name", "unknown");

  if ( !solver_name.compare("Hydro_Muscl_2D") or 
       !solver_name.compare("Hydro_Muscl_Block_2D")) {
    
    dimType = TWO_D;
    nbvar = 4;
    ghostWidth = 2;
    
  } else if ( !solver_name.compare("Hydro_Muscl_3D") or
              !solver_name.compare("Hydro_Muscl_Block_3D") ) {

    dimType = THREE_D;
    nbvar = 5;
    ghostWidth = 2;
    
  } else if ( !solver_name.compare("MHD_Muscl_2D") ) {

    dimType = TWO_D;
    nbvar = 8;
    ghostWidth = 3;
    mhdEnabled = true;
    
  } else if ( !solver_name.compare("MHD_Muscl_3D") ) {

    dimType = THREE_D;
    nbvar = 8;
    ghostWidth = 3;
    mhdEnabled = true;
    
  } else {

    // we should probably abort
    std::cerr << "Solver name not valid : " << solver_name << "\n";
    
  }
  
  /* initialize MESH parameters */
  nx = configMap.getInteger("mesh","nx", 1);
  ny = configMap.getInteger("mesh","ny", 1);
  nz = configMap.getInteger("mesh","nz", 1);

  xmin = configMap.getFloat("mesh", "xmin", 0.0);
  ymin = configMap.getFloat("mesh", "ymin", 0.0);
  zmin = configMap.getFloat("mesh", "zmin", 0.0);

  xmax = configMap.getFloat("mesh", "xmax", 1.0);
  ymax = configMap.getFloat("mesh", "ymax", 1.0);
  zmax = configMap.getFloat("mesh", "zmax", 1.0);

  boundary_type_xmin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_xmin", BC_ABSORBING));
  boundary_type_xmax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_xmax", BC_ABSORBING));
  boundary_type_ymin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_ymin", BC_ABSORBING));
  boundary_type_ymax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_ymax", BC_ABSORBING));
  boundary_type_zmin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_zmin", BC_ABSORBING));
  boundary_type_zmax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_zmax", BC_ABSORBING));

  level_min = configMap.getInteger("amr","level_min", 5);
  level_max = configMap.getInteger("amr","level_max", 10);

  // default value for amr_cycle_enabled
  bool amr_cycle_enabled_default = (level_min != level_max);

  // we can overwrite amr_cycle_enabled; e.g.
  // we can chose level_min != level_max for initial condition
  // but then swithoff amr cycle.
  amr_cycle_enabled = configMap.getBool("amr", "amr_cycle_enabled", amr_cycle_enabled_default);

  output_vtk_enabled  = configMap.getBool("output","vtk_enabled",true);
  output_hdf5_enabled = configMap.getBool("output","hdf5_enabled",false);

  settings.gamma0         = configMap.getFloat("hydro","gamma0", 1.4);
  settings.cfl            = configMap.getFloat("hydro", "cfl", 0.5);
  settings.iorder         = configMap.getInteger("hydro","iorder", 2);
  settings.slope_type     = configMap.getFloat("hydro","slope_type",1.0);
  settings.smallc         = configMap.getFloat("hydro","smallc", 1e-10);
  settings.smallr         = configMap.getFloat("hydro","smallr", 1e-10);

  // specific heat
  settings.cp             = configMap.getFloat("hydro", "cp", 0.0);

  // dynamic viscosity
  settings.mu             = configMap.getFloat("hydro", "mu", 0.0);

  // thermal diffusivity
  settings.kappa          = configMap.getFloat("hydro", "kappa", 0.0);
  
  niter_riemann  = configMap.getInteger("hydro","niter_riemann", 10);
  std::string riemannSolverStr = std::string(configMap.getString("hydro","riemann", "approx"));
  if ( !riemannSolverStr.compare("approx") ) {
    riemannSolverType = RIEMANN_APPROX;
  } else if ( !riemannSolverStr.compare("llf") ) {
    riemannSolverType = RIEMANN_LLF;
  } else if ( !riemannSolverStr.compare("hll") ) {
    riemannSolverType = RIEMANN_HLL;
  } else if ( !riemannSolverStr.compare("hllc") ) {
    riemannSolverType = RIEMANN_HLLC;
  } else if ( !riemannSolverStr.compare("hlld") ) {
    riemannSolverType = RIEMANN_HLLD;
  } else {
    std::cout << "Riemann Solver specified in parameter file is invalid\n";
    std::cout << "Use the default one : approx\n";
    riemannSolverType = RIEMANN_APPROX;
  }
  
  implementationVersion  = configMap.getFloat("OTHER","implementationVersion", 0);
  if (implementationVersion != 0 and
      implementationVersion != 1) {
    std::cout << "Implementation version is invalid (must be 0 or 1)\n";
    std::cout << "Use the default : 0\n";
    implementationVersion = 0;
  }

  std::string utype = configMap.getString("hydro", "updateType", "conservative_sum");
  if (utype == "conservative_sum")
    updateType = UPDATE_CONSERVATIVE_SUM;
  else
    updateType = UPDATE_NON_CONSERVATIVE;

  // low Mach correction
  rsst_enabled = configMap.getBool("low_mach", "rsst_enabled", false);
  rsst_cfl_enabled = configMap.getBool("low_mach", "rsst_cfl_enabled", false);
  rsst_ksi = configMap.getFloat("low_mach", "rsst_ksi", 10.0);

  // Gravity
  gravity_type = static_cast<GravityType>(configMap.getInteger("gravity", "gravity_type", GRAVITY_NONE));
  if (gravity_type & GRAVITY_CONSTANT) {
    gx = configMap.getFloat("gravity", "gx",  0.0);
    gy = configMap.getFloat("gravity", "gy", -1.0);
    gz = configMap.getFloat("gravity", "gz",  0.0);
  } 

  debug_output = configMap.getBool("output", "debug", false);

  init();
  setup_mpi(configMap);  
} // HydroParams::setup

// =======================================================
// =======================================================
void HydroParams::setup_mpi(ConfigMap& configMap)
{
  communicator = &dyablo::GlobalMpiSession::get_comm_world();
  // get world communicator size and check it is consistent with mesh grid sizes
  nProcs = communicator->MPI_Comm_size();

  // get my MPI rank inside topology
  myRank = communicator->MPI_Comm_rank();
  
} // HydroParams::setup_mpi

// =======================================================
// =======================================================
void HydroParams::init()
{

  // set other parameters
  imin = 0;
  jmin = 0;
  kmin = 0;

  imax = nx - 1 + 2*ghostWidth;
  jmax = ny - 1 + 2*ghostWidth;
  kmax = nz - 1 + 2*ghostWidth;
  
  isize = imax - imin + 1;
  jsize = jmax - jmin + 1;
  ksize = kmax - kmin + 1;

  dx = (xmax - xmin) / nx;
  dy = (ymax - ymin) / ny;
  dz = (zmax - zmin) / nz;
  
  settings.smallp  = settings.smallc*settings.smallc/
    settings.gamma0;
  settings.smallpp = settings.smallr*settings.smallp;
  settings.gamma6  = (settings.gamma0 + ONE_F)/(TWO_F * settings.gamma0);
  
  // check that given parameters are valid
  if ( (implementationVersion != 0) && 
       (implementationVersion != 1) && 
       (implementationVersion != 2) ) {
    fprintf(stderr, "The implementation version parameter should 0,1 or 2 !!!");
    fprintf(stderr, "Check your parameter file, section OTHER");
    exit(EXIT_FAILURE);
  } 

} // HydroParams::init


// =======================================================
// =======================================================
void HydroParams::print()
{
  
  printf( "##########################\n");
  printf( "Simulation run parameters:\n");
  printf( "##########################\n");
  //printf( "Solver name: %s\n",solver_name.c_str());
  printf( "nx         : %d\n", nx);
  printf( "ny         : %d\n", ny);
  printf( "nz         : %d\n", nz);
  
  printf( "dx         : %f\n", dx);
  printf( "dy         : %f\n", dy);
  printf( "dz         : %f\n", dz);

  printf( "imin       : %d\n", imin);
  printf( "imax       : %d\n", imax);

  printf( "jmin       : %d\n", jmin);      
  printf( "jmax       : %d\n", jmax);      

  printf( "kmin       : %d\n", kmin);      
  printf( "kmax       : %d\n", kmax);      

  printf( "ghostWidth : %d\n", ghostWidth);
  printf( "nbvar      : %d\n", nbvar);
  printf( "nStepmax   : %d\n", nStepmax);
  printf( "tEnd       : %f\n", tEnd);
  printf( "nOutput    : %d\n", nOutput);
  printf( "gamma0     : %f\n", settings.gamma0);
  printf( "gamma6     : %f\n", settings.gamma6);
  printf( "cfl        : %f\n", settings.cfl);
  printf( "smallr     : %12.10f\n", settings.smallr);
  printf( "smallc     : %12.10f\n", settings.smallc);
  printf( "smallp     : %12.10f\n", settings.smallp);
  printf( "smallpp    : %g\n", settings.smallpp);
  printf( "cp (specific heat)          : %g\n", settings.cp);
  printf( "mu (dynamic visosity)       : %g\n", settings.mu);
  printf( "kappa (thermal diffusivity) : %g\n", settings.kappa);
  //printf( "niter_riemann : %d\n", niter_riemann);
  printf( "iorder     : %d\n", settings.iorder);
  printf( "slope_type : %f\n", settings.slope_type);
  printf( "riemann    : %d\n", riemannSolverType);
  printf( "update type: %d\n", updateType);
  //printf( "problem    : %d\n", problemStr);
  printf( "implementation version : %d\n",implementationVersion);
  printf( "##########################\n");
  
} // HydroParams::print
