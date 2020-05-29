/**
 * \file test_CopyGhostBlockCellDataHash.cpp
 * \author Pierre Kestener
 * \date April, 26th 2020
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

#ifdef DYABLO_USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

#include "muscl/SolverHydroMuscl.h"
#include "muscl_block/SolverHydroMusclBlock.h"

#include "muscl_block/CopyInnerBlockCellData.h"
#include "muscl_block/CopyFaceBlockCellDataHash.h"
#include "muscl_block/ConvertToPrimitivesHydroFunctor.h"
#include "muscl_block/utils_block.h"

using Device = Kokkos::DefaultExecutionSpace;

namespace dyablo
{

// =======================================================================
// =======================================================================
void init_U(DataArrayBlock U,
            uint32_t nbOcts,
            HydroParams params,
            muscl_block::blockSize_t blockSizes,
            uint32_t ghostWidth,
            id2index_t fm)
{

  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  uint32_t nbTeams_ = 4;

  team_policy_t policy (nbTeams_,
                        Kokkos::AUTO() /* team size chosen by kokkos */);


  uint32_t bx = blockSizes[IX];
  uint32_t by = blockSizes[IY];
  uint32_t bz = blockSizes[IZ];

  uint32_t nbCells = params.dimType == TWO_D ?
    bx*by : bx*by*bz;

  Kokkos::parallel_for("init_U", policy,
                       KOKKOS_LAMBDA(const thread_t& member)
                       {
                         uint32_t iOct = member.league_rank();

                         while (iOct <  nbOcts) {

                           Kokkos::parallel_for(
                             Kokkos::TeamVectorRange(member, nbCells),
                             [&](const int32_t index) {
                               U(index, 0, iOct) =
                                 // something like a cellwise Morton index
                                 //index + iOct * nbCells;
                                 iOct;
                             }); // end TeamVectorRange

                           iOct += nbTeams_;

                         }

                       });

} // init_U

// =======================================================================
// =======================================================================
template<int dim>
void run_test()
{

  /*
   * testing CopyGhostBlockCellDataFunctor
   */
  std::cout << "// =============================================\n";
  std::cout << "// Testing CopyGhostBlockCellDataHashFunctor ...\n";
  std::cout << "// =============================================\n";

  /*
   * create a fake ConfigMap object
   */
  std::string input_file = dim==2 ? "dummy2d.ini" : "dummy3d.ini";
  ConfigMap configMap = broadcast_parameters(input_file);

  // test: create a HydroParams object
  HydroParams params = HydroParams();
  params.setup(configMap);

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

  muscl_block::blockSize_t blockSizes, blockSizes_g;
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

  uint32_t nbOctsPerGroup = 32;

  // ==================================================
  // STAGE 1 : PABLO object
  // ==================================================

  /*
   * initialize Pablo mesh
   *
   * 
   * 2d mesh should be exactly like this : 16 octants
   *
   *   ___________ __________
   *  |           |     |    |
   *  |           |  14 | 15 |
   *  |    11     |-----+----|
   *  |           |  12 | 13 |
   *  |___________|_____|____|
   *  |     |     | 8|9 |    |
   *  |  2  |  3  | 6|7 | 10 |
   *  |-----+-----|-----+----|
   *  |  0  |  1  |  4  | 5  |
   *  |_____|_____|_____|____|
   *  
   *
   */

  bitpit::PabloUniform amr_mesh(dim);

  // Set 2:1 balance
  // codim 1 ==> balance through faces
  // codim 2 ==> balance through faces and corner
  // codim 3 ==> balance through faces, edges and corner (3D only)
  int codim = 1; 
  amr_mesh.setBalanceCodimension(codim);
  
  uint32_t idx=0;
  amr_mesh.setBalance(idx,true);

  // set periodic border condition
  amr_mesh.setPeriodic(0);
  amr_mesh.setPeriodic(1);
  amr_mesh.setPeriodic(2);
  amr_mesh.setPeriodic(3);
  if (dim==3)
  {
    amr_mesh.setPeriodic(4);
    amr_mesh.setPeriodic(5);
  }

  // step 1
  amr_mesh.adaptGlobalRefine();
  amr_mesh.updateConnectivity();

  // step 2
  amr_mesh.setMarker(1,1);
  amr_mesh.adapt(true);
  amr_mesh.updateConnectivity();

  // step 3
  amr_mesh.setMarker(3,1);
  amr_mesh.adapt(true);
  amr_mesh.updateConnectivity();
  std::cout << "Mesh size is " << amr_mesh.getNumOctants() << "\n";

#if BITPIT_ENABLE_MPI==1
  // (Load)Balance the octree over the processes.
  amr_mesh.loadBalance();
#endif

  // ==================================================
  // STAGE 2 : create a AMRMetaData object
  // ==================================================

  uint64_t capacity = 1024*1024;
  AMRMetaData<dim> amrMetadata(capacity);

  amrMetadata.report();
  amrMetadata.update_hashmap(amr_mesh);
  amrMetadata.update_neigh_level_status(amr_mesh);
  amrMetadata.report();

  // ==================================================
  // STAGE 3 : fake user data
  // ==================================================

  // just retrieve a field manager
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);
  auto fm = fieldMgr.get_id2index();


  /*
   * allocate/initialize face data U
   */
  uint32_t nbOcts = amr_mesh.getNumOctants();
  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  DataArrayBlock U = DataArrayBlock("U", nbCellsPerOct, params.nbvar, nbOcts);

  init_U(U, nbOcts, params, blockSizes, ghostWidth, fm);

  uint32_t nghosts = amr_mesh.getNumGhosts();
  DataArrayBlock Ughost = DataArrayBlock("Ughost", nbCellsPerOct, params.nbvar, nghosts);

  /*
   * allocate/initialize Ugroup
   *
   * We assume total number of octant is sufficiently small to assume
   * a single group is enough.
   */

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";

  DataArrayBlock Ugroup = DataArrayBlock("Ugroup", nbCellsPerOct_g, params.nbvar, nbOctsPerGroup);
  uint32_t iGroup = 0;

  //uint8_t nfaces = (params.dimType == TWO_D ? 4 : 6);
  FlagArrayBlock Interface_flags = FlagArrayBlock("Interface Flags", nbOctsPerGroup);
  
  // first copy inner cells
  std::cout << "=========================================\n";
  std::cout << "Testing CopyInnerBlockCellDataFunctor....\n";
  std::cout << "=========================================\n";
  muscl_block::CopyInnerBlockCellDataFunctor::apply(configMap, 
                                                    params,
                                                    fm,
                                                    blockSizes,
                                                    ghostWidth, 
                                                    nbOcts,
                                                    nbOctsPerGroup,
                                                    U, 
                                                    Ugroup,
                                                    iGroup);

  std::cout << "=============================================\n";
  std::cout << "Testing CopyFaceBlockCellDataHashFunctor.....\n";
  std::cout << "=============================================\n";
  {
    muscl_block::CopyFaceBlockCellDataHashFunctor<dim>::apply(amrMetadata,
                                                              configMap,
                                                              params, 
                                                              fm,
                                                              blockSizes,
                                                              ghostWidth,
                                                              nbOctsPerGroup,
                                                              U, 
                                                              Ughost, 
                                                              Ugroup, 
                                                              iGroup,
                                                              Interface_flags);
    

    std::cout << "Copy Device data to host....\n";
    DataArrayBlock::HostMirror Ugroup_host =
      Kokkos::create_mirror(Ugroup);


    Kokkos::deep_copy(Ugroup_host, Ugroup);

    std::cout << "Start checking results....\n";
    for (uint32_t iOct = 0; iOct<nbOcts; ++iOct)
    {
      std::cout << "iOct=" << iOct << " // ==================================\n";

      if (bz>1) 
      {
      
        for (uint32_t iz = 0; iz < bz_g; ++iz)
        {
          for (uint32_t iy = 0; iy < by_g; ++iy)
          {
            for (uint32_t ix = 0; ix < bx_g; ++ix)
            {
              uint32_t index = ix + bx_g * (iy + by_g * iz);
              std::cout << std::right << std::setw(6) << Ugroup_host(index, 0, iOct) << " ";
            }
            std::cout << "\n";
          }
          std::cout << "\n";
        }
        
      } 
      else 
      {
        // //
        // // WARNING reverse order
        // //
        for (uint32_t iy2 = 0; iy2 < by_g; ++iy2)
        {
          uint32_t iy = by_g-1 - iy2;
          
          for (uint32_t ix = 0; ix < bx_g; ++ix)
          {
            uint32_t index = ix + bx_g * iy;
            std::cout << std::right << std::setw(6) << Ugroup_host(index, 0, iOct) << " ";
          }
          std::cout << "\n";
        }
        
      } // end if bz>1
    }

  } // end testing CopyFaceBlockCellDataFunctor


} // run_test

} // namespace dyablo

// =======================================================================
// =======================================================================
// =======================================================================
int main(int argc, char *argv[])
{

  // Create MPI session if MPI enabled
#ifdef DYABLO_USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // DYABLO_USE_MPI
  
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

#ifdef DYABLO_USE_MPI
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
#endif // DYABLO_USE_MPI
  }    // end kokkos config

  int testId = 2;
  if (argc>1)
    testId = std::atoi(argv[1]);

  // dim 2
  if (testId == 2)
    dyablo::run_test<2>();
  
  // dim3
  else if (testId == 3)
    dyablo::run_test<3>();

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
