#
# slighly refactored PABLO test for Kokkos user data
#

set(EXE_NAME PABLO_example_00008_kokkos)
add_executable(${EXE_NAME}
  PABLO_userDataComm.hpp
  PABLO_userDataComm.tpp
  PABLO_userDataLB.hpp
  PABLO_userDataLB.tpp
  PABLO_example_00008.cpp)
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
