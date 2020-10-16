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
# function to simplify creating a unit test
#
# Parameters:
#  - UNIT_TEST_NAME : the executable test name
#  - all other arg are expected to be source files
#
#####################################################################
function(dyablo_add_test UNIT_TEST_NAME)

  add_executable(${UNIT_TEST_NAME} "")
  target_sources(${UNIT_TEST_NAME} PRIVATE
    ${ARGN})

  # this is necessary to have BOOST_TEST_ALTERNATIVE_INIT_API
  # and thus being able to initialize with empty function init_function
  target_compile_definitions(${UNIT_TEST_NAME} PRIVATE BOOST_TEST_DYN_LINK)

  target_link_libraries(${UNIT_TEST_NAME} PUBLIC
    config
    Kokkos::kokkos
    dl
    )
  
  # muscl and muscl libraries only OK when using Kokkos::OpenMP for now
  if(NOT Kokkos_ENABLE_CUDA)
    target_link_libraries(${UNIT_TEST_NAME} PUBLIC
      muscl
      muscl_block
      )
  endif()

  target_link_libraries(${UNIT_TEST_NAME} PUBLIC
    shared
    monitoring
    BITPIT::BITPIT
    Boost::unit_test_framework)

  if (DYABLO_USE_MPI)
    target_compile_definitions(${UNIT_TEST_NAME} PUBLIC BITPIT_ENABLE_MPI)
    
    target_link_libraries(${UNIT_TEST_NAME} PUBLIC mpiUtils)
  endif()

  add_test(
    NAME ${UNIT_TEST_NAME}
    COMMAND ./${UNIT_TEST_NAME}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/unit_test/)
  
endfunction(dyablo_add_test)

#####################################################################
#
# here we create our tests
#
#####################################################################
dyablo_add_test(tst_shared_MortonUtils 
  shared/morton_utils_test.cpp
  main.cpp)

if(NOT Kokkos_ENABLE_CUDA)

# from old test/block_data
configure_file(
  block_data/test_implode_2D_block.ini
  block_data/test_implode_2D_block.ini COPYONLY)

configure_file(
  block_data/test_implode_3D_block.ini
  block_data/test_implode_3D_block.ini COPYONLY)

dyablo_add_test(utest_CopyInnerBlockCellData 
  block_data/test_CopyInnerBlockCellData.cpp
  main.cpp)

dyablo_add_test(utest_CopyGhostBlockCellData_2D 
  block_data/test_CopyGhostBlockCellData_2D.cpp
  main.cpp)

dyablo_add_test(utest_CopyGhostBlockCellData_3D 
  block_data/test_CopyGhostBlockCellData_3D.cpp
  main.cpp)

dyablo_add_test(utest_MusclBlockGodunovUpdateFunctor
  block_data/test_MusclBlockGodunovUpdateFunctor.cpp
  main.cpp)

dyablo_add_test(utest_CopyGhostBlockCellData_hash_openmp 
  block_data/test_CopyGhostBlockCellData_hash_openmp.cpp
  main.cpp)

# from old test/amr
configure_file(amr/dummy2d.ini amr/dummy2d.ini COPYONLY)
configure_file(amr/dummy3d.ini amr/dummy3d.ini COPYONLY)

dyablo_add_test(utest_AMRMetaData
  amr/test_AMRMetaData.cpp
  main.cpp)

endif()

dyablo_add_test(utest_LightOctree
  amr/test_LightOctree.cpp
  main.cpp)
  
  
  dyablo_add_test(utest_CopyGhostBlockCellDataHash
  amr/test_CopyGhostBlockCellDataHash.cpp
  main.cpp)


