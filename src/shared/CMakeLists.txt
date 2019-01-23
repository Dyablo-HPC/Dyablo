#######################
### SHARED ############
#######################
set (shared_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/problems/initRiemannConfig2d.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroParams.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroState.h
  ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/real_type.h
  ${CMAKE_CURRENT_SOURCE_DIR}/enums.h
  ${CMAKE_CURRENT_SOURCE_DIR}/EulerEquations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverBase.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/solver_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleVTKIO.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SimpleVTKIO.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bitpit_common.h
  )

add_library(shared
  ${shared_SRCS}
  )
if (USE_MPI)
  target_include_directories(shared
    PUBLIC
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/communications)
 
  target_compile_definitions(shared PUBLIC BITPIT_ENABLE_MPI)
endif (USE_MPI)

target_include_directories(shared
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_SOURCE_DIR}/src/utils/time
  ${CMAKE_SOURCE_DIR}/src/utils/io
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/PABLO
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/common
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/containers
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/IO
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/operators
  ) 
target_link_libraries(shared
  kokkos
  )

