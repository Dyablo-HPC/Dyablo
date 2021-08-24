#######################
### SHARED ############
#######################
add_library(shared "")

target_sources (shared
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleVTKIO.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh_hashmap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi/GhostCommunicator_pablo.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mpi/GhostCommunicator_kokkos.cpp
  )

if (DYABLO_USE_HDF5)
  target_sources(shared
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/HDF5_IO.cpp
    )
endif(DYABLO_USE_HDF5)

if (DYABLO_USE_MPI)
  target_compile_definitions(shared PUBLIC BITPIT_ENABLE_MPI)
endif (DYABLO_USE_MPI)

target_include_directories(shared
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/src/utils/time
  ${CMAKE_SOURCE_DIR}/src/utils/io)

target_link_libraries(shared 
  PUBLIC
    Kokkos::kokkos
    BITPIT::BITPIT
  )
if (DYABLO_USE_MPI)
  target_link_libraries(shared PUBLIC
    MPI::MPI_CXX
    )
endif(DYABLO_USE_MPI)

if(DYABLO_USE_HDF5)
  target_link_libraries(shared PUBLIC
    ${HDF5_LIBRARIES})
endif(DYABLO_USE_HDF5)
