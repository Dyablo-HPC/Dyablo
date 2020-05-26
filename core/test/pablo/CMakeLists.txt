macro(build_test_executable name with_kokkos)

  add_executable(${name} "")
  target_sources(${name}
    PUBLIC
    ${name}.cpp
    )
  
  # define how to build and link
  target_include_directories(${name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src)
  if(DYABLO_USE_MPI)
    target_compile_definitions(${name} PUBLIC BITPIT_ENABLE_MPI)
  endif(DYABLO_USE_MPI)
  target_link_libraries(${name} PUBLIC BITPIT::BITPIT)

  if (${with_kokkos})
    target_link_libraries(${name} PUBLIC kokkos)
  endif()
  
endmacro(build_test_executable)

set(KOKKOS_ENABLED TRUE)
set(KOKKOS_DISABLED FALSE)

build_test_executable(PABLO_test0       ${KOKKOS_DISABLED})
build_test_executable(PABLO_test1       ${KOKKOS_DISABLED})
build_test_executable(PABLO_test2       ${KOKKOS_DISABLED})
build_test_executable(mandelbrot        ${KOKKOS_DISABLED})
if (NOT Kokkos_ENABLE_CUDA)
  build_test_executable(mandelbrot_kokkos ${KOKKOS_ENABLED})
endif()

add_subdirectory(user_data)
