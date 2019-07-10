#
# euler_pablo test executable
#
set(EXE_NAME "test_solver")
add_executable(${EXE_NAME}
  test_solver.cpp
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


configure_file(test_four_quadrant_2D.ini test_four_quadrant_2D.ini COPYONLY)
configure_file(test_blast_2D.ini test_blast_2D.ini COPYONLY)
configure_file(test_blast_3D.ini test_blast_3D.ini COPYONLY)
configure_file(test_sod_2D.ini test_sod_2D.ini COPYONLY)
configure_file(test_isentropic_vortex_2D.ini test_isentropic_vortex_2D.ini COPYONLY)
configure_file(test_gresho_vortex_2D.ini test_gresho_vortex_2D.ini COPYONLY)
configure_file(test_kelvin_helmholtz_2D.ini test_kelvin_helmholtz_2D.ini COPYONLY)
configure_file(test_implode_2D.ini test_implode_2D.ini COPYONLY)
