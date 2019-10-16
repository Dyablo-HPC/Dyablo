macro(build_test_executable name with_kokkos)

  add_executable(${name} "")
  target_sources(${name}
    PUBLIC
    ${name}.cpp
    )
  
  # define how to build and link
  target_include_directories(${name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/common
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/containers
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/IO
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/operators
    ${CMAKE_SOURCE_DIR}/external/bitpit/src/PABLO)
  if(USE_MPI)
    target_include_directories(${name}
      PUBLIC
      ${CMAKE_SOURCE_DIR}/external/bitpit/src/communications)
    target_compile_definitions(${name} PUBLIC BITPIT_ENABLE_MPI)
  endif(USE_MPI)
  target_link_libraries(${name} PUBLIC ${BITPIT_LIBRARY})

  if (${with_kokkos})
    target_link_libraries(${name} PUBLIC kokkos ${BITPIT_LIBRARY})
  endif()
  
endmacro(build_test_executable)

set(KOKKOS_ENABLED TRUE)
set(KOKKOS_DISABLED FALSE)

build_test_executable(PABLO_test0       ${KOKKOS_DISABLED})
build_test_executable(PABLO_test1       ${KOKKOS_DISABLED})
build_test_executable(PABLO_test2       ${KOKKOS_DISABLED})
build_test_executable(mandelbrot        ${KOKKOS_DISABLED})
build_test_executable(mandelbrot_kokkos ${KOKKOS_ENABLED})

add_subdirectory(user_data)
