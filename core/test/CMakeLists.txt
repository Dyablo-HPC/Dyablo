add_subdirectory(kokkos)
add_subdirectory(pablo)

if (NOT Kokkos_ENABLE_CUDA)
  add_subdirectory(solver)
  add_subdirectory(convergence)
  add_subdirectory(block_data)
endif()

if (DYABLO_USE_HDF5)
  add_subdirectory(io)
endif(DYABLO_USE_HDF5)
