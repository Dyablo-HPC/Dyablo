#
# testing functors found in src/muscl_block subdirectory
#

add_executable(test_CopyInnerBlockCellData
  test_CopyInnerBlockCellData.cpp)
target_link_libraries(test_CopyInnerBlockCellData PUBLIC kokkos)
if (USE_MPI)
  target_link_libraries(test_CopyInnerBlockCellData
    PUBLIC 
    mpiUtils MPI::MPI_CXX)
endif(USE_MPI)

