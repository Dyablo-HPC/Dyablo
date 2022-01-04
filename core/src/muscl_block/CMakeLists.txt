set (muscl_block_SRCS
  # ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMusclBlock.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CopyGhostBlockCellData.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MapUserData.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MarkOctantsHydroFunctor.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitSod.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitBlast.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitImplode.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitGreshoVortex.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitFourQuadrant.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitIsentropicVortex.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitKelvinHelmholtz.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitShuOsher.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitDoubleMachReflection.cpp
  # ${CMAKE_CURRENT_SOURCE_DIR}/init/legacy/InitRayleighTaylor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_blast.cpp
  #${CMAKE_CURRENT_SOURCE_DIR}/update/MusclBlockUpdate_legacy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/update/MusclBlockUpdate_generic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_constant.cpp
  )

  if (DYABLO_USE_HDF5)
    list(APPEND muscl_block_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/io/IOManager_hdf5.cpp)
  endif(DYABLO_USE_HDF5)

add_library(muscl_block
  ${muscl_block_SRCS}
  )
#target_compile_options(muscl_block PUBLIC "-G")
target_include_directories(muscl_block
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src) 
target_link_libraries(muscl_block PUBLIC Kokkos::kokkos BITPIT::BITPIT)
target_link_libraries(muscl_block PUBLIC dyablo::utils::mpi)

if (DYABLO_USE_MPI)   
  target_compile_definitions(muscl_block PUBLIC BITPIT_ENABLE_MPI)
endif (DYABLO_USE_MPI)
