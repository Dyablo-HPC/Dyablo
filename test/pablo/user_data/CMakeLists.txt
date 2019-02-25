#
# slighly refactored PABLO test for Kokkos user data
#
macro(build_test_executable name)

  set(EXE_NAME "${name}")

  set (EXE_NAME_SRC
    PABLO_userDataComm.hpp
    PABLO_userDataComm.tpp
    PABLO_userDataLB.hpp
    PABLO_userDataLB.tpp
    ${name}.cpp
    )

  # define how to build and link
  add_executable(${EXE_NAME} ${EXE_NAME_SRC})
  target_include_directories(${EXE_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/common
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/containers
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/IO
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/operators
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/PABLO)
  if(USE_MPI)
    target_include_directories(${EXE_NAME}
      PUBLIC
      ${CMAKE_SOURCE_DIR}/external/bitpit/src/communications)
    target_compile_definitions(${EXE_NAME} PUBLIC BITPIT_ENABLE_MPI)
  endif(USE_MPI)
  target_link_libraries(${EXE_NAME} kokkos shared ${BITPIT_LIBRARY})

endmacro(build_test_executable)


build_test_executable(PABLO_example_00008_kokkos)
