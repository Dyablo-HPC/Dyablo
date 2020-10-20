add_subdirectory(kokkos)
add_subdirectory(pablo)
add_subdirectory(solver)

if (NOT Kokkos_ENABLE_CUDA)
  add_subdirectory(convergence)
endif()

if (DYABLO_USE_HDF5)
  add_subdirectory(io)
endif(DYABLO_USE_HDF5)
