# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)

# common solver code - TODO : make this work with CUDA backend
if (NOT Kokkos_ENABLE_CUDA)
  add_subdirectory(shared)

  # Hydro solver using 2nd order Muscl-Hancock solver
  add_subdirectory(muscl)
  add_subdirectory(muscl_block)
endif()

