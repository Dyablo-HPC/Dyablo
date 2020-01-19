#
# slighly refactored PABLO test for Kokkos user data
#
macro(build_test_executable name)

  add_executable(${name} "")
  target_sources(${name}
    PUBLIC
    PABLO_userDataComm.hpp
    PABLO_userDataComm.tpp
    PABLO_userDataLB.hpp
    PABLO_userDataLB.tpp
    ${name}.cpp
    )

  # define how to build and link
  target_include_directories(${name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/common
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/containers
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/IO
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/operators
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/PABLO)
  if(USE_MPI)
    target_include_directories(${name}
      PUBLIC
      ${CMAKE_SOURCE_DIR}/external/bitpit/src/communications)
    target_compile_definitions(${name} PUBLIC BITPIT_ENABLE_MPI)
  endif(USE_MPI)
  target_link_libraries(${name} kokkos shared ${BITPIT_LIBRARY})

endmacro(build_test_executable)


build_test_executable(PABLO_example_00006_kokkos)
build_test_executable(PABLO_example_00007_kokkos)
build_test_executable(PABLO_example_00008_kokkos)
