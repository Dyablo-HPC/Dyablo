#####################################################################
# Fetch Google test
#####################################################################

include(FetchContent)
FetchContent_Declare(
  googletest
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../external/gtest
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

#####################################################################
#
# function to simplify creating a unit test
#
# Parameters:
#  - UNIT_TEST_NAME : the executable test name
#  - all other arg are expected to be source files
#
#####################################################################
function(dyablo_add_test UNIT_TEST_NAME TEST_MPI)

  add_executable(${UNIT_TEST_NAME} "")
  target_sources(${UNIT_TEST_NAME} PRIVATE
    ${ARGN})

  target_link_libraries(${UNIT_TEST_NAME} PUBLIC
    dyablo_core
    config
    Kokkos::kokkos
    monitoring
    BITPIT::BITPIT
    gtest)

  if (DYABLO_USE_MPI)
    target_compile_definitions(${UNIT_TEST_NAME} PUBLIC BITPIT_ENABLE_MPI)
    
    target_link_libraries(${UNIT_TEST_NAME} PUBLIC mpiUtils)
  endif()

  gtest_add_tests(TARGET ${UNIT_TEST_NAME} 
                  EXTRA_ARGS "--gtest_output=xml:gtest_reports/" 
                  TEST_LIST ${UNIT_TEST_NAME}_TEST_LIST )
  set(${UNIT_TEST_NAME}_TEST_LIST ${${UNIT_TEST_NAME}_TEST_LIST} PARENT_SCOPE)

  # add_test(
  #   NAME ${UNIT_TEST_NAME}
  #   COMMAND ./${UNIT_TEST_NAME} --gtest_output=xml:gtest_reports/${UNIT_TEST_NAME}.xml
  #   WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/unit_test/)

  if(DYABLO_USE_MPI)
    if(TEST_MPI)
      add_test(
        NAME "${UNIT_TEST_NAME}_MPI"
        COMMAND cmake -E env
          OMP_NUM_THREADS=4 
          ${MPIEXEC_EXECUTABLE} -np 1 ./${UNIT_TEST_NAME} --gtest_output=xml:gtest_reports/${UNIT_TEST_NAME}_MPI.xml : -np 2 ./${UNIT_TEST_NAME}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/unit_test/)
    endif()
  endif()
  
endfunction(dyablo_add_test)

#####################################################################
#
# here we create our tests
#
#####################################################################
dyablo_add_test(utest_morton_utils OFF
  shared/test_morton_utils.cpp
  main.cpp)

#   if(NOT Kokkos_ENABLE_CUDA)
# dyablo_add_test(utest_CopyInnerBlockCellData OFF
#   block_data/test_CopyInnerBlockCellData.cpp
#   main.cpp)
#   endif()

# Import .ini files
configure_file(
  block_data/test_blast_2D_block.ini
  block_data/test_blast_2D_block.ini COPYONLY)
configure_file(
  block_data/test_blast_3D_block.ini
  block_data/test_blast_3D_block.ini COPYONLY)

dyablo_add_test(utest_AMRmesh ON
  amr/test_AMRmesh.cpp
  main.cpp)

dyablo_add_test(utest_LightOctree OFF
  amr/test_LightOctree.cpp
  main.cpp)

# dyablo_add_test(utest_CopyGhostBlockCellData ON 
#   block_data/test_CopyGhostBlockCellData.cpp
#   main.cpp)

# dyablo_add_test(utest_MusclBlockGodunovUpdateFunctor OFF
#   block_data/test_MusclBlockGodunovUpdateFunctor.cpp
#   main.cpp)

dyablo_add_test(utest_GhostCommunicator ON
  block_data/test_GhostCommunicator.cpp
  main.cpp)

dyablo_add_test(utest_LoadBalance ON
  block_data/test_LoadBalance.cpp
  main.cpp)

dyablo_add_test(utest_ForeachCell ON
  block_data/test_ForeachCell.cpp
  main.cpp)
  
dyablo_add_test(utest_MapUserData OFF
  block_data/test_MapUserData.cpp
  main.cpp)

dyablo_add_test(utest_GravitySolver_cg ON
  gravity/test_GravitySolver_cg.cpp
  main.cpp)

foreach( test "${utest_GravitySolver_cg_TEST_LIST}" )
  set_tests_properties(${test} PROPERTIES TIMEOUT 100) 
endforeach()
set_tests_properties(utest_GravitySolver_cg_MPI PROPERTIES TIMEOUT 100) 

# dyablo_add_test(utest_HydroConservativity ON
#   hydro/test_HydroConservativity.cpp
#   main.cpp)

dyablo_add_test(utest_States ON
  states/test_States.cpp
  main.cpp)

dyablo_add_test(utest_Kokkos_dyablo OFF
  test_Kokkos.cpp
  main.cpp)

# dyablo_add_test(utest_BoundaryConditions ON
#   boundary_conditions/utest_BoundaryConditions.cpp
#   main.cpp)

configure_file(
  hydro/test_blast_2D_block.ini
  hydro/test_blast_2D_block.ini COPYONLY)
configure_file(
  hydro/test_blast_3D_block.ini
  hydro/test_blast_3D_block.ini COPYONLY)

## Implementations registered to a factory that are contained in a separate static library have to be thourouhly tested
add_library( RegisteringFactory_impl_lib STATIC utils/RegisteringFactory_impl_to_register.cpp )
#target_link_libraries( RegisteringFactory_impl_lib PRIVATE shared )
dyablo_add_test(utest_RegisteringFactory OFF
  utils/test_RegisteringFactory.cpp
  main.cpp)
target_link_libraries( utest_RegisteringFactory PUBLIC RegisteringFactory_impl_lib )

dyablo_add_test(utest_MpiComm ON
  utils/test_MpiComm.cpp
  main.cpp)

dyablo_add_test(utest_Restart ON
  test_restart.cpp
  main.cpp)

dyablo_add_test(utest_UserData ON
  block_data/test_UserData.cpp
  main.cpp)
