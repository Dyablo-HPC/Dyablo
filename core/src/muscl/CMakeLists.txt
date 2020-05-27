set (muscl_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitImplode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitSod.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitBlast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitImplode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitGreshoVortex.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitFourQuadrant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitIsentropicVortex.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitKelvinHelmholtz.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitRayleighTaylor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitShuOsher.cpp
  )

# add MPI user data communication interface
if (DYABLO_USE_MPI)
  list(APPEND muscl_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/UserDataComm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UserDataLB.cpp)
endif(DYABLO_USE_MPI)

add_library(muscl
  ${muscl_SRCS}
  )
target_include_directories(muscl
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ) 
target_link_libraries(muscl PUBLIC Kokkos::kokkos BITPIT::BITPIT)

if (DYABLO_USE_MPI)
 
  target_compile_definitions(muscl PUBLIC BITPIT_ENABLE_MPI)

  target_link_libraries(muscl PUBLIC MPI::MPI_CXX)

endif (DYABLO_USE_MPI)
