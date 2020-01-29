add_subdirectory(kokkos)

if (NOT Kokkos_ENABLE_CUDA)
  add_subdirectory(pablo)
  add_subdirectory(solver)
  add_subdirectory(convergence)
  add_subdirectory(block_data)
endif()

if (USE_HDF5)
  add_subdirectory(io)
endif(USE_HDF5)
