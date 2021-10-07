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
  muscl
  config
  Kokkos::kokkos
  dl
  muscl_block
  shared
  monitoring
  BITPIT::BITPIT)

if (DYABLO_USE_MPI)  
  target_compile_definitions(${EXE_NAME} PUBLIC BITPIT_ENABLE_MPI)

  target_link_libraries(${EXE_NAME} mpiUtils)
endif()

add_backward(${EXE_NAME})

# Files for gitlab-ci testcases
# Blast 3D block based 
configure_file(test_blast_3D_block.ini test_blast_3D_block.ini COPYONLY)
configure_file(visu_blast_3D_block.pvsm visu_blast_3D_block.pvsm COPYONLY)
# Blast 3D cell based 
configure_file(test_blast_3D.ini test_blast_3D.ini COPYONLY)
configure_file(visu_blast_3D_cell.pvsm visu_blast_3D_cell.pvsm COPYONLY)
# Blast 2D block based
configure_file(test_blast_2D_block.ini test_blast_2D_block.ini COPYONLY)
configure_file(visu_blast_2D_block.pvsm visu_blast_2D_block.pvsm COPYONLY)
# Blast 2D cell based 
configure_file(test_blast_2D.ini test_blast_2D.ini COPYONLY)
configure_file(visu_blast_2D_cell.pvsm visu_blast_2D_cell.pvsm COPYONLY)
