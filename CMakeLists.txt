# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.16)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.19.0")
  message(STATUS "Setting policy CMP0114 to use new behavior")
  cmake_policy(SET CMP0114 OLD)
endif()

###################################################################
# Use the CMake Superbuild pattern to help build dependencies 
# before building Dyablo.
# Uses ExternalProject_add to build dependencies :
# - bitpit
# (Kokkos is compiled as a subdyrectory in Dyablo)
# external dependencies are also verified
# - bitpit : libxml2, MPI
# - Dyablo : MPI, HDF5
# Dependencies may be verified twice (here and when compilig Dyablo externalproject)
# The intent is to find as much external dependency errors at configure time as possible
###################################################################
project (dyablo_superbuild LANGUAGES C CXX)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/core/cmake)
include(preventBuildInSource)

###################################################################
# Global options for Dyablo supported by Superbuild
# Additional options can be accessed from actual Dyablo project core/CMakeLists.txt
###################################################################
option (DYABLO_ENABLE_UNIT_TESTING "Enable unit testing" OFF)
option (DYABLO_USE_MPI "Activate / want MPI build" ON)
option (DYABLO_USE_HDF5 "build HDF5 input/output support" ON)
option (DYABLO_USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)
option (DYABLO_USE_PABLO "Use AMRmesh_pablo of AMR" OFF )
option (DYABLO_USE_INTERNAL_BITPIT "Compile bitpit from source in external/bitpit" ON)
set( DYABLO_CXX_COMPILER  ${CMAKE_CXX_COMPILER} CACHE STRING "compiler used to build dyablo")
set( BITPIT_COMPILER  ${CMAKE_CXX_COMPILER} CACHE STRING "compiler used to build bitpit when DYABLO_USE_INTERNAL_BITPIT=ON")
option ( DYABLO_HIDE_BITPIT_COMPILATION "Don't show bitpit compilation log in console" ON )
option (DYABLO_USE_INTERNAL_KOKKOS "Use internal Kokkos from external/kokkos" ON)
option (Kokkos_ENABLE_CUDA "Enable Kokkos::CUDA backend" OFF)
option (Kokkos_ENABLE_SYCL "Enable Kokkos::SYCL backend" OFF)
set( Kokkos_ARCH  "" CACHE STRING "List of architectures to set for Kokkos : -DKokkos_ARCH_<arch>=ON")
set( DYABLO_CMAKE_ARGS "" CACHE STRING "Additional cmake command line arguments to forward to Dyablo subproject"  )

include (ExternalProject)
set (DYABLO_EXTERNALPROJECT_DEPENDENCIES)
set (DYABLO_CMAKE_ARGS_ ${DYABLO_CMAKE_ARGS})

if( DYABLO_USE_PABLO )
  if( DYABLO_USE_INTERNAL_BITPIT )
  message( STATUS "== Setup Bitpit ExternalProject ==" )

    # verify bitpit dependencies
    find_package( LibXml2 REQUIRED )
    if(DYABLO_USE_MPI)
      find_package( MPI REQUIRED COMPONENTS C CXX )
    endif()

    find_package(Git REQUIRED)

    set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external)
    ExternalProject_Add (bitpit_external
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/bitpit
      UPDATE_COMMAND ""
      CMAKE_ARGS
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -DCMAKE_CXX_COMPILER=${BITPIT_COMPILER}
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DMPI_C_COMPILER=${MPI_C_COMPILER}
        -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
        -DLIBXML2_LIBRARY=${LIBXML2_LIBRARY}
        -DBITPIT_ENABLE_INTEGRATION_TESTS=OFF
        -DBITPIT_ENABLE_MPI=${DYABLO_USE_MPI}
        -DBITPIT_MODULE_LA=OFF
        -DBITPIT_MODULE_CG=OFF
        -DBITPIT_MODULE_DISCRETIZATION=OFF
        -DBITPIT_MODULE_LEVELSET=OFF
        -DBITPIT_MODULE_PATCHKERNEL=OFF
        -DBITPIT_MODULE_POD=OFF
        -DBITPIT_MODULE_RBF=OFF
        -DBITPIT_MODULE_SA=OFF
        -DBITPIT_MODULE_SURFUNSTRUCTURED=OFF
        -DBITPIT_MODULE_VOLCARTESIAN=OFF
        -DBITPIT_MODULE_VOLOCTREE=OFF
        -DBITPIT_MODULE_VOLUNSTRUCTURED=OFF
        -DBITPIT_MODULE_POINTCLOUD=OFF
        -DBITPIT_MODULE_LINEUNSTRUCTURED=OFF
      LOG_CONFIGURE ${DYABLO_HIDE_BITPIT_COMPILATION}
      LOG_BUILD ${DYABLO_HIDE_BITPIT_COMPILATION}
      LOG_INSTALL ${DYABLO_HIDE_BITPIT_COMPILATION}
    )

    list( APPEND DYABLO_EXTERNALPROJECT_DEPENDENCIES bitpit_external )
    set( BITPIT_VERSION_SHORT "1.9" )
    set( DYABLO_BITPIT_DIR "${CMAKE_BINARY_DIR}/external/Install/bitpit_external/lib/cmake/bitpit-${BITPIT_VERSION_SHORT}" )
  else( DYABLO_USE_INTERNAL_BITPIT )
    # Find installed bitpit
    find_package(BITPIT CONFIG REQUIRED)
    set( DYABLO_BITPIT_DIR ${BITPIT_DIR} )
  endif()
endif()

if( DYABLO_USE_INTERNAL_KOKKOS )
  # Read list from ${Kokkos_ARCH} and set appropriate -DKokkos_ARCH_*=ON
  message( STATUS "Forced architectures for Kokkos:" )
  foreach(ARCH ${Kokkos_ARCH})
    list (APPEND DYABLO_CMAKE_ARGS_
      -DKokkos_ARCH_${ARCH}=ON
    )
    message( STATUS "- Kokkos_ARCH_${ARCH}=ON" )
  endforeach()
  set(DYABLO_KOKKOS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos")
else( DYABLO_USE_INTERNAL_KOKKOS )
  # Find installed Kokkos 
  find_package(Kokkos 3.3 REQUIRED)

  set(DYABLO_KOKKOS_DIR ${Kokkos_DIR})
endif()

#####################################
# Now we build dyablo
#####################################

message( STATUS "== Setup Dyablo ExternalProject ==" )

# Check external dependencies
if( DYABLO_USE_MPI )
  find_package(MPI REQUIRED)
  list (APPEND DYABLO_CMAKE_ARGS_
    -DMPI_C_COMPILER=${MPI_C_COMPILER}
    -DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}
  )
endif()
# hdf5
if (DYABLO_USE_HDF5)
  set( HDF5_PREFER_PARALLEL ON )
  find_package( HDF5 REQUIRED COMPONENTS C)
  if( DYABLO_USE_MPI AND NOT HDF5_IS_PARALLEL )
    message( FATAL_ERROR "HDF5 library must be parallel when compiling Dyablo with MPI" )
  endif()
  list (APPEND DYABLO_CMAKE_ARGS_
    -DHDF5_C_COMPILER_EXECUTABLE=${HDF5_C_COMPILER_EXECUTABLE}
  )
endif()

# default arguments - common to all platform
list (APPEND DYABLO_CMAKE_ARGS_
  -DCMAKE_CXX_COMPILER=${DYABLO_CXX_COMPILER}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DDYABLO_ENABLE_UNIT_TESTING=${DYABLO_ENABLE_UNIT_TESTING}
  -DDYABLO_USE_PABLO=${DYABLO_USE_PABLO}
  -DDYABLO_USE_MPI=${DYABLO_USE_MPI}
  -DDYABLO_USE_HDF5=${DYABLO_USE_HDF5}
  -DDYABLO_USE_MPI_CUDA_AWARE_ENFORCED=${DYABLO_USE_MPI_CUDA_AWARE_ENFORCED}
  -DDYABLO_USE_INTERNAL_KOKKOS=${DYABLO_USE_INTERNAL_KOKKOS}
  -DKokkos_ENABLE_CUDA=${Kokkos_ENABLE_CUDA}
  -DKokkos_ENABLE_SYCL=${Kokkos_ENABLE_SYCL}
  -DBITPIT_DIR=${DYABLO_BITPIT_DIR}
  -DKokkos_DIR=${DYABLO_KOKKOS_DIR}
)

# Build dyablo
ExternalProject_Add (dyablo
  DEPENDS ${DYABLO_EXTERNALPROJECT_DEPENDENCIES}
  BUILD_ALWAYS TRUE
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/core
  CMAKE_ARGS ${DYABLO_CMAKE_ARGS_}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dyablo)

# add target dyablo-test
if (DYABLO_ENABLE_UNIT_TESTING)
  ExternalProject_Add_Step (dyablo test
    COMMAND           make test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dyablo
    COMMENT           "running dyablo unit tests"
    ALWAYS            TRUE
    EXCLUDE_FROM_MAIN TRUE
    )
  ExternalProject_Add_StepTargets(dyablo test)
endif()

message(  ${DYABLO_CMAKE_ARGS_} )

message( WARNING "This is not an error, just a reminder: 
CMake superbuild is meant to help you setup dyablo by compiling dependencies (PABLO, etc...) before configuring the actual Dyablo cmake project.
CMake variables are not forwarded to the actual Dyablo CMake project : if you wish to setup dyablo with custom variables (such as CMAKE_CXX_FLAGS), you need to modify the CMakeCache (with cmake-gui or ccmake for instance) from subdirectory ${CMAKE_CURRENT_BINARY_DIR}/dyablo" )

if( CMAKE_CXX_FLAGS )
  message( FATAL_ERROR "CMAKE_CXX_FLAGS is not forwarded to Dyablo subproject : see warning above" )
endif()