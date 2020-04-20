#
# AMR tests / examples
#

add_executable(test_AMRMetaData
  test_AMRMetaData.cpp)
target_link_libraries(test_AMRMetaData 
  PUBLIC 
  shared
  kokkos
  BITPIT::BITPIT)

if (USE_MPI)
  target_link_libraries(test_AMRMetaData 
    PUBLIC 
    mpiUtils MPI::MPI_CXX)
endif(USE_MPI)

