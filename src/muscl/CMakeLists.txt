set (muscl_SRCS
  # ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor2D.h
  # ${CMAKE_CURRENT_SOURCE_DIR}/HydroBaseFunctor3D.h
  # ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors2D.h
  # ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunFunctors3D.h
  # ${CMAKE_CURRENT_SOURCE_DIR}/HydroInitFunctors2D.h
  # ${CMAKE_CURRENT_SOURCE_DIR}/HydroInitFunctors3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SolverHydroMuscl.cpp
  )

add_library(muscl
  ${muscl_SRCS}
  )
target_include_directories(muscl
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/kokkos/core/src
  ${CMAKE_SOURCE_DIR}/kokkos/containers/src
  ${CMAKE_SOURCE_DIR}/kokkos/algorithms/src
  ${CMAKE_BINARY_DIR}/kokkos
  ) 
target_link_libraries(muscl
  kokkos
  )
