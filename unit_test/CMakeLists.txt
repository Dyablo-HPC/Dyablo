add_library(dyablo-tests-driver STATIC main.cpp)
target_include_directories(dyablo-tests-driver PRIVATE
  ${Boost_INCLUDE_DIRS})


set(UNIT_TEST_NAME "dyablo_unit_tests")

add_executable(${UNIT_TEST_NAME} "")
target_sources(${UNIT_TEST_NAME} PRIVATE
  shared/morton_utils_test.cpp)
target_link_libraries(${UNIT_TEST_NAME} PRIVATE
  config
  kokkos
  dl
  muscl
  muscl_block
  shared
  monitoring
  BITPIT::BITPIT
  dyablo-tests-driver)

if (USE_MPI)
  target_compile_definitions(${UNIT_TEST_NAME} PUBLIC BITPIT_ENABLE_MPI)

  target_link_libraries(${UNIT_TEST_NAME} PRIVATE mpiUtils)
endif()

add_test(NAME ${UNIT_TEST_NAME}
  COMMAND ${UNIT_TEST_NAME})
