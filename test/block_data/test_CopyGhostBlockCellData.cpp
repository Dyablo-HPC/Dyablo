/**
 * \file test_CopyGhostBlockCellData.cpp
 * \author Pierre Kestener
 * \date September, 24th 2019
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>


#include "shared/real_type.h"    // choose between single and double precision
#include "shared/HydroParams.h"  // read parameter file
#include "shared/solver_utils.h" // print monitoring information
#include "shared/FieldManager.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

#include "muscl/SolverHydroMuscl.h"
#include "muscl_block/SolverHydroMusclBlock.h"

#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyFaceBlockCellData.h"
#include "muscl_block/ConvertToPrimitivesHydroFunctor.h"

using Device = Kokkos::DefaultExecutionSpace;

namespace dyablo
{

namespace muscl_block
{

// =======================================================================
// =======================================================================
void run_test(int argc, char *argv[])
{

  /*
   * testing CopyGhostBlockCellDataFunctor
   */
  std::cout << "// =========================================\n";
  std::cout << "// Testing CopyGhostBlockCellDataFunctor ...\n";
  std::cout << "// =========================================\n";

  /*
   * read parameter file and initialize a ConfigMap object
   */
  // only MPI rank 0 actually reads input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "test_implode_2D_block.ini";
  ConfigMap configMap = broadcast_parameters(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

  // retrieve solver name from settings
  const std::string solver_name = configMap.getString("run", "solver_name", "Unknown");

  // actually initializing a solver
  // initialize workspace memory (U, U2, ...)
  if (solver_name.find("Muscl_Block") == std::string::npos)
  {

    std::cerr << "Please modify your input parameter file.\n";
    std::cerr << "  solver name must contain string \"Muscl_Block\"\n";

  }

  // anyway create the right solver
  SolverHydroMusclBlock* solver = new SolverHydroMusclBlock(params, configMap);

  // by now, init condition must have been called

  // just retrieve a field manager
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);

  auto fm = fieldMgr.get_id2index();

  /*
   * "geometry" setup
   */

  // block ghost width
  uint32_t ghostWidth = configMap.getInteger("amr", "ghostwidth", 2);

  // block sizes
  uint32_t bx = configMap.getInteger("amr", "bx", 0);
  uint32_t by = configMap.getInteger("amr", "by", 0);
  uint32_t bz = configMap.getInteger("amr", "bz", 1);

  // block sizes with ghosts
  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = by + 2 * ghostWidth;
  uint32_t bz_g = bz + 2 * ghostWidth;

  blockSize_t blockSizes, blockSizes_g;
  blockSizes[IX] = bx;
  blockSizes[IY] = by;
  blockSizes[IZ] = bz;

  blockSizes_g[IX] = bx_g;
  blockSizes_g[IY] = by_g;
  blockSizes_g[IZ] = bz_g;

  std::cout << "Using " 
            << "bx=" << bx << " "
            << "by=" << by << " "
            << "bz=" << bz << " "
            << "bx_g=" << bx_g << " "
            << "by_g=" << by_g << " "
            << "bz_g=" << bz_g << " "
            << "ghostwidth=" << ghostWidth << "\n";

  uint32_t nbCellsPerOct = params.dimType == TWO_D ? bx * by : bx * by * bz;
  uint32_t nbCellsPerOct_g =
      params.dimType == TWO_D ? bx_g * by_g : bx_g * by_g * bz_g;

  uint32_t nbOctsPerGroup = configMap.getInteger("amr", "nbOctsPerGroup", 32);

  /*
   * allocate/initialize Ugroup
   */

  uint32_t nbOcts = solver->amr_mesh->getNumOctants();
  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";

  DataArrayBlock Ugroup = DataArrayBlock("Ugroup", nbCellsPerOct_g, params.nbvar, nbOctsPerGroup);
  uint32_t iGroup = 1;

  uint8_t nfaces = (params.dimType == TWO_D ? 4 : 6);
  FlagArrayBlock Interface_flags = FlagArrayBlock("Interface Flags", nbOctsPerGroup);
  
  // chose an octant which should have a "same size" neighbor in all direction
  //uint32_t iOct_local = 2;
  
  // chose an octant which should have at least
  // a "larger size" neighbor in one direction
  //uint32_t iOct_local = 30;

  // chose an octant which should have at least
  // an interface with "smaller size" neighbor in one direction
  uint32_t iOct_local = 26;

  uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;

  std::cout << "Looking at octant id = " << iOct_global << "\n";

  // octant location
  double x = solver->amr_mesh->getX(iOct_global);
  double y = solver->amr_mesh->getY(iOct_global);
  std::cout << "Octant location : x=" << x << " y=" << y << "\n";

  // save solution, just for cross-checking
  solver->save_solution();

  std::cout << "Printing U data from iOct = " << iOct_global << "\n";
  for (uint32_t iz=0; iz<bz; ++iz) 
  {
    for (uint32_t iy=0; iy<by; ++iy)
    {
      for (uint32_t ix=0; ix<bx; ++ix)
      {
        uint32_t index = ix + bx*(iy+by*iz);
        printf("%5f ",solver->U(index,fm[ID],iOct_global));
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  std::cout << "Ugroup sizes = " 
            << Ugroup.extent(0) << " "
            << Ugroup.extent(1) << " "
            << Ugroup.extent(2) << "\n";

  // first copy inner cells

  //uint32_t nbOcts = solver->amr_mesh->getNumOctants();
  DataArrayBlock Gravity("Gravity", nbCellsPerOct, 3, nbOcts);
  DataArrayBlock Gravity_host = Kokkos::create_mirror(Gravity);
  DataArrayBlock Ggroup("Ggroup", nbCellsPerOct_g, 3, nbOctsPerGroup);
  DataArrayBlock Gravity_ghost;

  CopyInnerBlockCellDataFunctor::apply(configMap, params, fm, 
                                       blockSizes,
                                       ghostWidth, 
                                       nbOcts,
                                       nbOctsPerGroup,
                                       solver->U, Ugroup, Gravity, Ggroup, iGroup);

  std::cout << "==========================================";
  std::cout << "Testing CopyFaceBlockCellDataFunctor....\n";
  {
    CopyFaceBlockCellDataFunctor::apply(solver->amr_mesh,
                                        configMap,
                                        params, 
                                        fm,
                                        blockSizes,
                                        ghostWidth,
                                        nbOctsPerGroup,
                                        solver->U, 
                                        solver->Ughost, 
                                        Gravity, Gravity_ghost, Ggroup,
                                        Ugroup, 
                                        iGroup,
                                        Interface_flags);
    
    // print data from from the chosen iGroup 
    std::cout << "Printing Ugroup data from iOct = " << iOct_global << " | iOctLocal = " << iOct_local << " and iGroup = " << iGroup << "\n";
    if (bz>1) 
    {
      
      for (uint32_t iz = 0; iz < bz_g; ++iz)
      {
        for (uint32_t iy = 0; iy < by_g; ++iy)
        {
          for (uint32_t ix = 0; ix < bx_g; ++ix)
          {
            uint32_t index = ix + bx_g * (iy + by_g * iz);
            printf("%5f ", Ugroup(index, fm[ID], iOct_local));
          }
          std::cout << "\n";
        }
        std::cout << "\n";
      }
      
    } 
    else 
    {
      
      for (uint32_t iy = 0; iy < by_g; ++iy)
      {
        for (uint32_t ix = 0; ix < bx_g; ++ix)
        {
          uint32_t index = ix + bx_g * iy;
          printf("%5f ", Ugroup(index, fm[IP], iOct_local));
        }
        std::cout << "\n";
      }
      
    } // end if bz>1

  } // end testing CopyFaceBlockCellDataFunctor

  // also testing ConvertToPrimitivesHydroFunctor
  std::cout << "==========================================";
  std::cout << "Testing ConvertToPrimitivesHydroFunctor \n";
  {
    DataArrayBlock Qgroup = DataArrayBlock("Qgroup", nbCellsPerOct_g, params.nbvar, nbOctsPerGroup);

    uint32_t nbOcts = solver->amr_mesh->getNumOctants();
    
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

    for (uint32_t iy = 0; iy < by_g; ++iy)
    {
      for (uint32_t ix = 0; ix < bx_g; ++ix)
      {
        uint32_t index = ix + bx_g * iy;
        printf("%5f ", Qgroup(index, fm[IP], iOct_local));
      }
      std::cout << "\n";
    }

  } // end testing ConvertToPrimitivesHydroFunctor

  delete solver;

} // run_test

} // namespace muscl_block

} // namespace dyablo

// =======================================================================
// =======================================================================
// =======================================================================
int main(int argc, char *argv[])
{

  // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI
  
  Kokkos::initialize(argc, argv);

  int rank = 0;
  int nRanks = 1;

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
# ifdef KOKKOS_ENABLE_CUDA
    {

      // To enable kokkos accessing multiple GPUs don't forget to
      // add option "--ndevices=X" where X is the number of GPUs
      // you want to use per node.

      // on a large cluster, the scheduler should assign ressources
      // in a way that each MPI task is mapped to a different GPU
      // let's cross-checked that:
      
      int cudaDeviceId;
      cudaGetDevice(&cudaDeviceId);
      std::cout << "I'm MPI task #" << rank << " (out of " << nRanks << ")"
		<< " pinned to GPU #" << cudaDeviceId << "\n";
      
    }
# endif // KOKKOS_ENABLE_CUDA
#endif // USE_MPI
  }    // end kokkos config

  dyablo::muscl_block::run_test(argc, argv);

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
