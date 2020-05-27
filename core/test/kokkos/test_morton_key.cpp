/**
 * This executable is used to test Morton key routines
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <cstdint>

#include <bitset>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"
#include "shared/enums.h"

#include "shared/morton_utils.h"

namespace dyablo
{

/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
void test_morton_2d()
{

  std::cout << "===========\n";
  std::cout << "=====2D====\n";
  std::cout << "===========\n";

  uint32_t ix = 15;
  uint32_t iy = 7;

  uint64_t ix_s = splitBy3<2>(ix);
  uint64_t iy_s = splitBy3<2>(iy);

  std::cout << "Coord: " << ix << "," << iy << "\n";
  std::cout << "splitBy3<2>(" << ix << "=" << std::bitset<4>(ix) << ")="
	    << ix_s  << "=" << std::bitset<12>(ix_s) << "\n";
  std::cout << "splitBy3<2>(" << iy << "=" << std::bitset<4>(iy) << ")="
	    << iy_s  << "=" << std::bitset<12>(iy_s) << "\n";

  std::cout << "Morton index of (" << ix << "," << iy << ") = " << compute_morton_key(ix,iy)
	    << "=" << std::bitset<8>(compute_morton_key(ix,iy)) << "\n";


  uint64_t key = compute_morton_key(ix,iy);
  std::cout << "Reverse morton key:\n";
  std::cout << "key = " << key << " ==> extracted x,y = "
	    << morton_extract_bits<2,IX>(key) << ","
	    << morton_extract_bits<2,IY>(key)
	    << "\n";
  
} // test_morton_2d


void test_morton_3d()
{

  std::cout << "===========\n";
  std::cout << "=====3D====\n";
  std::cout << "===========\n";

  uint32_t ix = 15;
  uint64_t ix_s = splitBy3<3>(ix);
  uint32_t iy = 7;
  uint64_t iy_s = splitBy3<3>(iy);
  uint32_t iz = 2;
  uint64_t iz_s = splitBy3<3>(iz);

  std::cout << "splitBy3<3>(" << ix << "=" << std::bitset<4>(ix) << ")="
	    << ix_s  << "=" << std::bitset<12>(ix_s) << "\n";
  std::cout << "splitBy3<3>(" << iy << "=" << std::bitset<4>(iy) << ")="
	    << iy_s  << "=" << std::bitset<12>(iy_s) << "\n";
  std::cout << "splitBy3<3>(" << iz << "=" << std::bitset<4>(iz) << ")="
	    << iz_s  << "=" << std::bitset<12>(iz_s) << "\n";

  std::cout << "Morton index of (" << ix << "," << iy << "," << iz << ") = "
	    << compute_morton_key(ix,iy,iz)
	    << "=" << std::bitset<12>(compute_morton_key(ix,iy,iz)) << "\n";

  uint64_t key = compute_morton_key(ix,iy,iz);
  std::cout << "Reverse morton key:\n";
  std::cout << "key = " << key << " ==> extracted x,y,z = "
	    << morton_extract_bits<3,IX>(key) << ","
	    << morton_extract_bits<3,IY>(key) << ","
	    << morton_extract_bits<3,IZ>(key)
	    << "\n";

} // test_morton_3d

} // namespace dyablo

/*************************************************/
/*************************************************/
/*************************************************/
int main(int argc, char* argv[])
{

  Kokkos::initialize(argc, argv);

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
  }


  // instantiate some tests
  dyablo::test_morton_2d();
  dyablo::test_morton_3d();

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
} // end main
