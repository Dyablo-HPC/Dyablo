#
# testing functors found in src/muscl_block subdirectory
#

add_executable(test_CopyInnerBlockCellData
  test_CopyInnerBlockCellData.cpp)
target_link_libraries(test_CopyInnerBlockCellData PUBLIC 
  config
  kokkos
  shared 
  monitoring)
if (USE_MPI)
  target_link_libraries(test_CopyInnerBlockCellData
    PUBLIC 
    mpiUtils MPI::MPI_CXX)
endif(USE_MPI)

# some useful parameter file
configure_file(test_implode_2D_block.ini test_implode_2D_block.ini COPYONLY)
