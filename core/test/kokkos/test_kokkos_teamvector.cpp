/**
 * \file test_kokkos_teamvector.cpp
 * \author Pierre Kestener
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

#ifdef DYABLO_USE_MPI
#include "utils/mpi/GlobalMpiSession.h"
#include <mpi.h>
#endif // DYABLO_USE_MPI

using Device = Kokkos::DefaultExecutionSpace;

using Data_t = Kokkos::View<int32_t *, Device>;
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

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  /**
   * test parallel for functor
   *
   */
  TestKokkosTeamVectorForFunctor(Data_t data, uint32_t bSize)
      : data(data), bSize(bSize) {
    nbBlocks = (data.extent(0) + bSize - 1) / bSize;
  };

  // static method which does it all: create and execute functor
  static void apply(Data_t data, uint32_t bSize) {

    TestKokkosTeamVectorForFunctor functor(data, bSize);

    // kokkos execution policy
    uint32_t nbTeams_ = 16;
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_,
                         Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("TestKokkosTeamVectorForFunctor", policy, functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const {

    uint32_t iBlock = member.league_rank();

    while (iBlock < nbBlocks) {

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, bSize),
          KOKKOS_LAMBDA(const int32_t index) {
            // copy q state in q global
            data(index + iBlock * bSize) += 12;
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

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Kokkos team vector - parallel reduce inside block
 *
 * \note in this test, data array size must be a multiple of nbBlocks
 */
class TestKokkosTeamVectorReduceFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  /**
   * test parallel reduce functor
   *
   */
  TestKokkosTeamVectorReduceFunctor(Data_t data, uint32_t bSize)
      : data(data), bSize(bSize) {
    nbBlocks = (data.extent(0) + bSize - 1) / bSize;
  };

  // static method which does it all: create and execute functor
  static void apply(Data_t data, uint32_t bSize) {

    TestKokkosTeamVectorReduceFunctor functor(data, bSize);

    // kokkos execution policy
    uint32_t nbTeams_ = 16;
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_,
                         Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("TestKokkosTeamVectorReduceFunctor", policy,
                         functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const {

    uint32_t iBlock = member.league_rank();

    while (iBlock < nbBlocks) {

      int sum=0;

      Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, bSize),
          KOKKOS_LAMBDA(const int32_t index, int& local_sum) {
            // copy q state in q global
            local_sum += data(index + iBlock * bSize);
          },
          sum); // end TeamVectorRange

      // check results :
      int32_t diff = sum - (bSize-1)*bSize/2 - iBlock*bSize*bSize;
      bool valid = diff==0 ? true : false;

      //std::cout << iBlock << ": res=" << sum << " results valid ? " << valid << "\n";
      printf("%d: res=%d results valid ? %d\n",iBlock,sum,valid);

      iBlock += nbTeams;

    } // end while iBlock < nbBlocks

  } // operator

  //! heavy data
  Data_t data;

  //! block size
  uint32_t bSize;

  //! number of blocks
  uint32_t nbBlocks;

}; // TestKokkosTeamVectorReduceFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Kokkos team vector - parallel reduce.
 *
 * Do a hierarchical reduce, i.e. first reduce inside a team, then a global reduce.
 *
 * \note in this test, data array size must be a multiple of nbBlocks
 */
class TestKokkosTeamVectorReduceFunctor2 {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  /**
   * test parallel reduce functor
   *
   */
  TestKokkosTeamVectorReduceFunctor2(Data_t data, uint32_t bSize)
      : data(data), bSize(bSize) {
    nbBlocks = (data.extent(0) + bSize - 1) / bSize;
  };

  // static method which does it all: create and execute functor
  static void apply(Data_t data, uint32_t bSize) {

    TestKokkosTeamVectorReduceFunctor2 functor(data, bSize);

    // kokkos execution policy
    uint32_t nbTeams_ = 16;
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_,
                         Kokkos::AUTO() /* team size chosen by kokkos */);

    // initialize reduce value to something really small
    int32_t max_value = std::numeric_limits<int32_t>::min();

    Kokkos::parallel_reduce("TestKokkosTeamVectorReduceFunctor2", policy,
                            functor,
                            max_value);

    printf("maximum value is %d\n",max_value);

  }

  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile int32_t& dst,
             const volatile int32_t& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  // ====================================================================
  // ====================================================================
  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (int32_t& dst) const
  {

  // initialize to something really small
#ifdef __CUDA_ARCH__
    dst = -0x7f800000;
#else
    dst = std::numeric_limits<int32_t>::min();
#endif // __CUDA_ARCH__
    
  } // init

  KOKKOS_INLINE_FUNCTION
  void operator()(thread_t member, int32_t &max_value) const {

    uint32_t iBlock = member.league_rank();
    uint32_t index = member.team_rank();

    int32_t result = max_value;

    while (iBlock < nbBlocks) {

      while (index < bSize) {

        //printf("[debug] index=%d iBlock=%d bSize=%d | data=%d\n",index,iBlock,bSize,data(index + iBlock * bSize));

        // update result
        result = 
          data(index + iBlock * bSize) > result ?
                                         data(index + iBlock * bSize) : result;

        index += member.team_size();
        
      }

      iBlock += nbTeams;

    } // end while iBlock < nbBlocks

    // update global reduced value
    if (max_value < result)
      max_value = result;

  } // operator

  //! heavy data
  Data_t data;

  //! block size
  uint32_t bSize;

  //! number of blocks
  uint32_t nbBlocks;

}; // TestKokkosTeamVectorReduceFunctor2

// =======================================================================
// =======================================================================
void run_test(uint32_t bSize, uint32_t nbBlocks) {

  /*
   * TestKokkosTeamVectorForFunctor
   */
  {
    std::cout << "// ======================================\n";
    std::cout << "Testing TestKokkosTeamVectorForFunctor...\n";
    std::cout << "// ======================================\n";

    uint32_t dataSize = bSize * nbBlocks;

    // create and init test data
    Data_t data = Data_t("test_data", dataSize);
    Kokkos::parallel_for(
        "init_test_data", dataSize, KOKKOS_LAMBDA(uint32_t i) { data(i) = i; });

    // Kokkos::fence();
    // Kokkos::parallel_for("print_results", dataSize,
    //   KOKKOS_LAMBDA(uint32_t i) {
    //                        std::cout << i << " " << data(i) << "\n";
    //   });

    TestKokkosTeamVectorForFunctor::apply(data, bSize);

    Kokkos::fence();

    Kokkos::parallel_for(
        "print_results", dataSize, KOKKOS_LAMBDA(uint32_t i) {
          //std::cout << i << " " << data(i) << "\n";
          printf("%d %d\n",i,data(i));
        });
  }

  /*
   * TestKokkosTeamVectorReduceFunctor
   */
  {
    std::cout << "// ======================================\n";
    std::cout << "Testing TestKokkosTeamVectorReduceFunctor...\n";
    std::cout << "// ======================================\n";

    uint32_t dataSize = bSize * nbBlocks;

    // create and init test data
    Data_t data = Data_t("test_data", dataSize);
    Kokkos::parallel_for(
        "init_test_data", dataSize, KOKKOS_LAMBDA(uint32_t i) { data(i) = i; });

    // Kokkos::fence();
    // Kokkos::parallel_for("print_results", dataSize,
    //   KOKKOS_LAMBDA(uint32_t i) {
    //                        std::cout << i << " " << data(i) << "\n";
    //   });

    TestKokkosTeamVectorReduceFunctor::apply(data, bSize);

  }

  /*
   * TestKokkosTeamVectorReduceFunctor2 - this i a "max" reduction
   */
  {
    std::cout << "// ======================================\n";
    std::cout << "Testing TestKokkosTeamVectorReduceFunctor2...\n";
    std::cout << "// ======================================\n";

    uint32_t dataSize = bSize * nbBlocks;

    // create and init test data
    Data_t data = Data_t("test_data", dataSize);
    Kokkos::parallel_for(
        "init_test_data", dataSize, KOKKOS_LAMBDA(uint32_t i) { 
          data(i) = 12-(i-13)*(i-15); 
        });

    TestKokkosTeamVectorReduceFunctor2::apply(data, bSize);

  }
  
} // run_test

// =======================================================================
// =======================================================================
int main(int argc, char *argv[]) {

  // Create MPI session if MPI enabled
  dyablo::GlobalMpiSession mpiSession(&argc, &argv);

  Kokkos::initialize(argc, argv);

  int rank = 0;
  int nRanks = 1;

  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";

    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if (Kokkos::hwloc::available()) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE[" << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT[" << Kokkos::hwloc::get_available_threads_per_core()
          << "] )" << std::endl;
    }
    Kokkos::print_configuration(msg);
    std::cout << msg.str();
    std::cout << "##########################\n";

#ifdef DYABLO_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#ifdef KOKKOS_ENABLE_CUDA
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
#endif // DYABLO_USE_MPI
  }    // end kokkos config

  uint32_t bSize = 4;
  uint32_t nbBlocks = 32;
  run_test(bSize, nbBlocks);

  Kokkos::finalize();

  return EXIT_SUCCESS;
}
