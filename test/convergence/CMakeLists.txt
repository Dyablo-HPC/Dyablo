#
# Space convergence for MUSCL scheme
#
set(EXE_NAME "muscl_convergence")
add_executable(${EXE_NAME}
  muscl_convergence.cpp
  ${BACKWARD_ENABLE}
  )

target_include_directories(${EXE_NAME}
  PUBLIC
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/common
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/containers
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/IO
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/operators
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/PABLO
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/src
  )

target_link_libraries(${EXE_NAME}
  config
  kokkos
  dl
  muscl
  shared
  monitoring
  ${BITPIT_LIBRARY})

if (USE_MPI)
  target_include_directories(${EXE_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/communications)
  
  target_compile_definitions(${EXE_NAME} PUBLIC BITPIT_ENABLE_MPI)

  target_link_libraries(${EXE_NAME} mpiUtils)
endif()

add_backward(${EXE_NAME})

#configure_file(test_isentropic_vortex_2D.ini test_isentropic_vortex_2D.ini COPYONLY)
