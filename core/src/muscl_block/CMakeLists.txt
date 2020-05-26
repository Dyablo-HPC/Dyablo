set (muscl_block_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMusclBlock.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitSod.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitBlast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitImplode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitGreshoVortex.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitFourQuadrant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitIsentropicVortex.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitKelvinHelmholtz.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitShuOsher.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitDoubleMachReflection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitRayleighTaylor.cpp
  )

# add MPI user data communication interface
if (USE_MPI)
  list(APPEND muscl_block_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/UserDataComm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UserDataLB.cpp)
endif(USE_MPI)

add_library(muscl_block
  ${muscl_block_SRCS}
  )
target_include_directories(muscl_block
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src) 
target_link_libraries(muscl_block PUBLIC kokkos BITPIT::BITPIT)

if (USE_MPI)
   
  target_compile_definitions(muscl_block PUBLIC BITPIT_ENABLE_MPI)

  target_link_libraries(muscl_block PUBLIC MPI::MPI_CXX)

endif (USE_MPI)
