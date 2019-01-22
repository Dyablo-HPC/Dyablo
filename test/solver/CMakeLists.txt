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

add_backward(${EXE_NAME})

if (USE_MPI)
  target_link_libraries(${EXE_NAME} mpiUtils)
endif()


configure_file(test_four_quadrant_2D.ini test_four_quadrant_2D.ini COPYONLY)
