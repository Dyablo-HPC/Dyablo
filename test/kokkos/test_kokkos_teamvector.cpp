/**
 * \file test_kokkos_teamvector.cpp
 * \author Pierre Kestener
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI


using Device = Kokkos::DefaultExecutionSpace;

using Data_t = Kokkos::View<int32_t*, Device>;
using DataHost_t = Data_t::HostMirror;

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Kokkos team vector - parallel for
 *
 * \note in this test, data array size must be a multiple of nbBlocks
 */
class TestKokkosTeamVectorForFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  /**
   * test parallel for functor
   *
   */
  TestKokkosTeamVectorForFunctor(Data_t data, uint32_t bSize) :
    data(data),
    bSize(bSize)
    {
      nbBlocks = (data.extent(0) + bSize -1)/bSize;
    };

  // static method which does it all: create and execute functor
  static void apply(Data_t data, uint32_t bSize)
  {

    TestKokkosTeamVectorForFunctor functor(data,bSize);
    
    // kokkos execution policy
    uint32_t nbTeams_ = 16; 
    functor.setNbTeams ( nbTeams_ );
    
    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);
    
    Kokkos::parallel_for("TestKokkosTeamVectorForFunctor",
                         policy, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const
  {
    
    uint32_t iBlock = member.league_rank();

    while (iBlock < nbBlocks)
    {

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, bSize),
        KOKKOS_LAMBDA(const int32_t index) {
          
          // copy q state in q global
          data(index+iBlock*bSize) += 12;

        }); // end TeamVectorRange

      iBlock += nbTeams;
      
    } // end while iBlock < nbBlocks

  } // operator
  
  //! heavy data
  Data_t data;

  //! block size
  uint32_t bSize;

  //! number of blocks
  uint32_t nbBlocks;

}; // TestKokkosTeamVectorForFunctor

// =======================================================================
// =======================================================================
void run_test(uint32_t bSize, uint32_t nbBlocks)
{

  uint32_t dataSize = bSize*nbBlocks;

  // create and init test data
  Data_t data = Data_t("test_data",dataSize);
  Kokkos::parallel_for("init_test_data", dataSize,
    KOKKOS_LAMBDA(uint32_t i) {
                         data(i) = i;
    });

  // Kokkos::fence();
  // Kokkos::parallel_for("print_results", dataSize,
  //   KOKKOS_LAMBDA(uint32_t i) {
  //                        std::cout << i << " " << data(i) << "\n";
  //   });

  TestKokkosTeamVectorForFunctor::apply(data, bSize);

  Kokkos::fence();

  Kokkos::parallel_for("print_results", dataSize,
    KOKKOS_LAMBDA(uint32_t i) {
                         std::cout << i << " " << data(i) << "\n";
    });

} // run_test

// =======================================================================
// =======================================================================
int main(int argc, char* argv[])
{

 // Create MPI session if MPI enabled
#ifdef USE_MPI
  hydroSimu::GlobalMpiSession mpiSession(&argc,&argv);
#endif // USE_MPI

  Kokkos::initialize(argc, argv);

  int rank=0;
  int nRanks=1;

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
#endif // KOKKOS_ENABLE_CUDA
#endif // USE_MPI
  } // end kokkos config

  uint32_t bSize = 8;
  uint32_t nbBlocks = 12;
  run_test(bSize, nbBlocks);

  Kokkos::finalize();

  return EXIT_SUCCESS;

}
