#
# AMR tests / examples
#

# ======================================================
# ======================================================
add_executable(test_AMRMetaData
  test_AMRMetaData.cpp)
target_link_libraries(test_AMRMetaData 
  PUBLIC 
  shared
  Kokkos::kokkos
  BITPIT::BITPIT)

if (DYABLO_USE_MPI)
  target_link_libraries(test_AMRMetaData 
    PUBLIC 
    mpiUtils MPI::MPI_CXX)
endif(DYABLO_USE_MPI)

# ======================================================
# ======================================================
add_executable(test_CopyGhostBlockCellDataHash
  test_CopyGhostBlockCellDataHash.cpp)
target_link_libraries(test_CopyGhostBlockCellDataHash PUBLIC
  config
  Kokkos::kokkos
  dl
  shared
  monitoring
  BITPIT::BITPIT)
if (DYABLO_USE_MPI)
  target_link_libraries(test_CopyGhostBlockCellDataHash
    PUBLIC
    mpiUtils MPI::MPI_CXX)
endif(DYABLO_USE_MPI)

configure_file(dummy2d.ini dummy2d.ini COPYONLY)
configure_file(dummy3d.ini dummy3d.ini COPYONLY)
