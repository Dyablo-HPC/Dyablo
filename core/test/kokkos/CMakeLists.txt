#
# Kokkos tests / examples
#

add_executable(query_device_kokkos
  query_device_kokkos.cpp)
target_link_libraries(query_device_kokkos PUBLIC kokkos)
if (USE_MPI)
  target_link_libraries(query_device_kokkos PUBLIC MPI::MPI_CXX)
endif(USE_MPI)

add_executable(test_kokkos_teamvector
  test_kokkos_teamvector.cpp)
target_link_libraries(test_kokkos_teamvector PUBLIC kokkos)
if (USE_MPI)
  target_link_libraries(test_kokkos_teamvector PUBLIC mpiUtils MPI::MPI_CXX)
endif(USE_MPI)

##############################################
add_executable(test_unordered_map
  test_unordered_map.cpp)
target_include_directories(test_unordered_map
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_unordered_map kokkos hwloc dl)

