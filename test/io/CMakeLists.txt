macro(build_test_executable name with_kokkos)

  add_executable(${name} "")
  target_sources(${name}
    PUBLIC
    ${name}.cpp
    )
  
  # define how to build and link
  target_include_directories(${name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src)
  if(USE_MPI)
    target_compile_definitions(${name} PUBLIC BITPIT_ENABLE_MPI)
  endif(USE_MPI)
  target_link_libraries(${name} PUBLIC BITPIT::BITPIT)

  if (${with_kokkos})
    target_link_libraries(${name} PUBLIC kokkos)
  endif()

  target_link_libraries(${name} PUBLIC shared dyablo::config)
  if (USE_MPI)
    target_link_libraries(${name} PUBLIC mpiUtils)
  endif(USE_MPI)

endmacro(build_test_executable)

set(KOKKOS_ENABLED TRUE)
set(KOKKOS_DISABLED FALSE)

# all the tests are only meant to run on host but we
# should later be able to run them when Kokkos::Cuda
# execution space is enable
#
# STATUS :
# - currently, only test_vtk runs OK when Kokkos::Cuda activated
#
# TODO :
# - refactor HDF5 writer class to use the host mirror data as input;
#   this will make the test_hdf5 run with Kokkos::Cuda activated
# - refactor muscl_block init functors to make test_hdf5_block run 
#   with Kokkos::Cuda activated
#

build_test_executable(test_vtk        ${KOKKOS_ENABLED})
build_test_executable(test_hdf5_block ${KOKKOS_ENABLED})
build_test_executable(test_hdf5       ${KOKKOS_ENABLED})

configure_file(test_io_vtk.ini  test_io_vtk.ini  COPYONLY)
configure_file(test_io_hdf5.ini test_io_hdf5.ini COPYONLY)
configure_file(test_io_hdf5_block.ini test_io_hdf5_block.ini COPYONLY)
