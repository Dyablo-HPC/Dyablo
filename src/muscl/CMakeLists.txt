set (muscl_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/init/HydroInitFunctors.h
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitSod.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ComputeDtHydroFunctor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ConvertToPrimitivesHydroFunctor.h)

# add MPI user data communication interface
if (USE_MPI)
  list(APPEND muscl_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/UserDataComm.h
    ${CMAKE_CURRENT_SOURCE_DIR}/UserDataComm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UserDataLB.h
    ${CMAKE_CURRENT_SOURCE_DIR}/UserDataLB.cpp)
endif(USE_MPI)

add_library(muscl
  ${muscl_SRCS}
  )
target_include_directories(muscl
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/PABLO
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/common
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/containers
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/IO
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/operators
  ) 
target_link_libraries(muscl PUBLIC kokkos)

if (USE_MPI)
  
  target_include_directories(muscl
    PUBLIC
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/communications)
 
  target_compile_definitions(muscl PUBLIC BITPIT_ENABLE_MPI)

  target_link_libraries(muscl PUBLIC MPI::MPI_CXX)

endif (USE_MPI)
