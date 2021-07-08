#
# Kokkos tests / examples
#

add_executable(query_device_kokkos
  query_device_kokkos.cpp)
target_link_libraries(query_device_kokkos PUBLIC Kokkos::kokkos)
if (DYABLO_USE_MPI)
  target_link_libraries(query_device_kokkos PUBLIC MPI::MPI_CXX)
endif(DYABLO_USE_MPI)

add_executable(test_kokkos_teamvector
  test_kokkos_teamvector.cpp)
target_link_libraries(test_kokkos_teamvector PUBLIC Kokkos::kokkos)
if (DYABLO_USE_MPI)
  target_link_libraries(test_kokkos_teamvector PUBLIC mpiUtils MPI::MPI_CXX)
endif(DYABLO_USE_MPI)

##############################################
add_executable(test_unordered_map
  test_unordered_map.cpp)
target_include_directories(test_unordered_map
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_unordered_map Kokkos::kokkos)

##############################################
add_executable(test_morton_key
  test_morton_key.cpp)
target_include_directories(test_morton_key
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_morton_key Kokkos::kokkos)

##############################################
add_executable(compute_mandelbrot_amr
  compute_mandelbrot_amr.cpp)
target_include_directories(compute_mandelbrot_amr
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
target_link_libraries(compute_mandelbrot_amr Kokkos::kokkos io)
if(DYABLO_USE_MPI)
  target_link_libraries(compute_mandelbrot_amr
    mpiUtils)
endif(DYABLO_USE_MPI)

##############################################
add_executable(test_AMRBlockForeachCell_ghost
  test_AMRBlockForeachCell_ghost.cpp)
target_include_directories(test_AMRBlockForeachCell_ghost
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_AMRBlockForeachCell_ghost PUBLIC shared monitoring mpiUtils config muscl_block)

add_executable(test_AMRBlockForeachCell_local
test_AMRBlockForeachCell_local.cpp)
target_include_directories(test_AMRBlockForeachCell_local
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(test_AMRBlockForeachCell_local PUBLIC monitoring mpiUtils shared muscl_block )

