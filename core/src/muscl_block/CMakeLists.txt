set (muscl_block_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_blast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/update/MusclBlockUpdate_generic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_generic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_second_derivative_error.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_pseudo_gradient.cpp
  )

list(APPEND muscl_block_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/legacy/MapUserData.cpp 
  ${CMAKE_CURRENT_SOURCE_DIR}/legacy/CopyGhostBlockCellData.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/legacy/MarkOctantsHydroFunctor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/update/MusclBlockUpdate_legacy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_legacy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_legacy.cpp
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
