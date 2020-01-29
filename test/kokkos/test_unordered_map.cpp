/**
 * This executable is used to test Kokkos::UnorderedMap
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <cstdint>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"
#include "shared/enums.h"

#include <Kokkos_UnorderedMap.hpp>

using DataMap = Kokkos::UnorderedMap<uint64_t, uint64_t, dyablo::Device>;

/**
 * Functor to fill a Kokkos::UnorderedMap
 */
template<int dim>
struct fill_map
{

  DataMap dataMap;
  int init_size;

  fill_map( DataMap i_dataMap, int i_init_size ) :
    dataMap(i_dataMap),
    init_size(i_init_size)
  {
    Kokkos::parallel_for(init_size, *this);
  }

  /*
   * 2D version.
   */
  //! functor for 2d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==2, int>::type& i) const
  {
    uint64_t key = (uint64_t) i;
    uint64_t value = 2*key;
    dataMap.insert( key, value );
  } 
  
  /*
   * 3D version.
   */
  //! functor for 3d 
  template<int dim_ = dim>
  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::Impl::enable_if<dim_==3, int>::type& i) const
  {
    uint64_t key = (uint64_t) i;
    uint64_t value = 3*key;
    dataMap.insert( key, value );
  } 
  
}; // struct fill_map


/*
 *
 * Main test using scheme order as template parameter.
 * order is the number of solution points per direction.
 *
 */
void test_unordered_map_2d(int N)
{

  std::cout << "===========\n";
  std::cout << "=====2D====\n";
  std::cout << "===========\n";

  DataMap dataMap(2*N*N);
  
  std::cout << "dataMap.size()     = " << dataMap.size() << std::endl;
  std::cout << "dataMap.capacity() = " << dataMap.capacity() << " (max size)" << std::endl;

  fill_map<2> fill(dataMap, N*N);

  std::cout << "After fill_map\n";
  std::cout << "dataMap.size()     = " << dataMap.size() << std::endl;
  std::cout << "dataMap.capacity() = " << dataMap.capacity() << " (max size)" << std::endl;

  DataMap::HostMirror dataMapOnHost(dataMap.capacity());
  Kokkos::deep_copy(dataMapOnHost, dataMap);
  
  for (std::size_t i=0; i<dataMapOnHost.capacity(); ++i) {
    if (dataMapOnHost.valid_at(i)) {
      std::cout << i << " "
		<< "dataMapOnHost["
		<< dataMapOnHost.key_at(i) << "]=" 
		<< dataMapOnHost.value_at(i) << "\n";
    }
  }

  // resize hashmap
  dataMapOnHost.rehash(dataMap.capacity()*2);

  // print again (should be unchanged)
  std::cout << "Print again after rehash.... (key_at have changed)\n";
  for (std::size_t i=0; i<dataMapOnHost.capacity(); ++i) {
    if (dataMapOnHost.valid_at(i)) {
      std::cout << i << " "
		<< "dataMapOnHost["
		<< dataMapOnHost.key_at(i) << "]=" 
		<< dataMapOnHost.value_at(i) << "\n";
    }
  }

  // print using Kokkos
  std::cout << "Printing dataMap using Kokkos::Impl::UnorderedMapPrint:\n";
  Kokkos::Impl::UnorderedMapPrint<DataMap> printer(dataMap);
  printer.apply();
  
} // test_unordered_map_2d<N>


void test_unordered_map_3d(int N)
{

  std::cout << "===========\n";
  std::cout << "=====3D====\n";
  std::cout << "===========\n";
    
  DataMap dataMap(2*N*N*N);

  std::cout << "dataMap.size()     = " << dataMap.size() << std::endl;
  std::cout << "dataMap.capacity() = " << dataMap.capacity() << " (max size)" << std::endl;

  fill_map<3> fill(dataMap, N*N*N);

  std::cout << "After fill_map\n";
  std::cout << "dataMap.size()     = " << dataMap.size() << std::endl;
  std::cout << "dataMap.capacity() = " << dataMap.capacity() << " (max size)" << std::endl;

  DataMap::HostMirror dataMapOnHost(dataMap.capacity());
  Kokkos::deep_copy(dataMapOnHost, dataMap);

  for (std::size_t i=0; i<dataMapOnHost.capacity(); ++i) {
    if (dataMapOnHost.valid_at(i)) {
      std::cout << i << " "
		<< "dataMapOnHost["
		<< dataMapOnHost.key_at(i) << "]=" 
		<< dataMapOnHost.value_at(i) << "\n";
    }
  }

} // test_unordered_map_3d<N>

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
  test_unordered_map_2d(5);

  test_unordered_map_3d(4);

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
}
