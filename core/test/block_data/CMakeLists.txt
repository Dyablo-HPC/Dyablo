#
# testing functors found in src/muscl_block subdirectory
#

# ======================================================
# ======================================================
add_executable(test_CopyInnerBlockCellData
  test_CopyInnerBlockCellData.cpp)
target_link_libraries(test_CopyInnerBlockCellData PUBLIC 
  config
  Kokkos::kokkos
  shared 
  monitoring)
if (DYABLO_USE_MPI)
  target_link_libraries(test_CopyInnerBlockCellData
    PUBLIC 
    mpiUtils MPI::MPI_CXX)
endif(DYABLO_USE_MPI)

# ======================================================
# ======================================================
add_executable(test_CopyGhostBlockCellData
  test_CopyGhostBlockCellData.cpp)
target_link_libraries(test_CopyGhostBlockCellData PUBLIC 
  config
  Kokkos::kokkos
  dl
  muscl_block
  shared
  monitoring
  BITPIT::BITPIT)
if (DYABLO_USE_MPI)
  target_link_libraries(test_CopyGhostBlockCellData
    PUBLIC 
    mpiUtils MPI::MPI_CXX)
endif(DYABLO_USE_MPI)

# ======================================================
# ======================================================
add_executable(test_CopyGhostBlockCellData_hash_openmp
  test_CopyGhostBlockCellData_hash_openmp.cpp)
target_link_libraries(test_CopyGhostBlockCellData_hash_openmp PUBLIC 
  config
  kokkos
  dl
  muscl_block
  shared
  monitoring
  BITPIT::BITPIT)
if (USE_MPI)
  target_link_libraries(test_CopyGhostBlockCellData_hash_openmp
    PUBLIC 
    mpiUtils MPI::MPI_CXX)
endif(USE_MPI)

# ======================================================
# ======================================================
add_executable(test_MusclBlockGodunovUpdateFunctor
  test_MusclBlockGodunovUpdateFunctor.cpp)
target_link_libraries(test_MusclBlockGodunovUpdateFunctor PUBLIC 
  config
  Kokkos::kokkos
  dl
  muscl_block
  shared
  monitoring
  BITPIT::BITPIT)
if (DYABLO_USE_MPI)
  target_link_libraries(test_MusclBlockGodunovUpdateFunctor
    PUBLIC 
    mpiUtils MPI::MPI_CXX)
endif(DYABLO_USE_MPI)

# some useful parameter file
configure_file(test_implode_2D_block.ini test_implode_2D_block.ini COPYONLY)
