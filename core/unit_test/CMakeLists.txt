#####################################################################
# Boost (for unit testing)
# Note : we only check for boost if unit testing is enabled
#####################################################################
find_package(Boost 1.65 REQUIRED COMPONENTS unit_test_framework)

message(" BOOST found : ${Boost_FOUND}")
if (Boost_FOUND)
  message(" BOOST version   : ${Boost_VERSION}")
  message(" BOOST inc dirs  : ${Boost_INCLUDE_DIRS}")
  message(" BOOST lib dirs  : ${Boost_LIBRARY_DIRS}")
  message(" BOOST libraries : ${Boost_LIBRARIES}")
endif(Boost_FOUND)

#####################################################################
#
# macro to simplify creating a unit test
#
# fist arg is the test name
# all other arg are expected to be source files
#
#####################################################################
macro(dyablo_add_test UNIT_TEST_NAME)

  #set(UNIT_TEST_NAME "dyablo_unit_tests")

  add_executable(${UNIT_TEST_NAME} "")
  target_sources(${UNIT_TEST_NAME} PRIVATE
    ${ARGN} main.cpp)

  # this is necessary to have BOOST_TEST_ALTERNATIVE_INIT_API
  # and thus being able to initialize with empty function init_function
  target_compile_definitions(${UNIT_TEST_NAME} PRIVATE BOOST_TEST_DYN_LINK)
  target_link_libraries(${UNIT_TEST_NAME} PRIVATE
    config
    Kokkos::kokkos
    dl
    shared
    monitoring
    BITPIT::BITPIT
    Boost::unit_test_framework)
  
  # muscl and muscl libraries only OK when using Kokkos::OpenMP for now
  if(NOT Kokkos_ENABLE_CUDA)
    target_link_libraries(${UNIT_TEST_NAME} PRIVATE
      muscl
      muscl_block
      )
  endif()

  if (DYABLO_USE_MPI)
    target_compile_definitions(${UNIT_TEST_NAME} PUBLIC BITPIT_ENABLE_MPI)
    
    target_link_libraries(${UNIT_TEST_NAME} PRIVATE mpiUtils)
  endif()

  add_test(NAME ${UNIT_TEST_NAME} COMMAND ./${UNIT_TEST_NAME})
  
endmacro(dyablo_add_test)

#####################################################################
#
# here we create our tests
#
#####################################################################
dyablo_add_test(tst_shared_MortonUtils shared/morton_utils_test.cpp)


