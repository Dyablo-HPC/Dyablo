# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.10)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
  message(STATUS "Setting policy CMP0076 to use new behavior")
  cmake_policy(SET CMP0076 NEW)
endif()

###################################################################
#
# Handle superbuild first building 
# 1. required dependencies (here only bitpit)
# 2. bitpit + kokkos
#
# To clarify : the superbuild drives the build of
# - bitpit
# - dyablo + kokkos
# both using the cmake ExternalProject_add macro.
#
# The main reason behind this design is that it allows to separate in a 
# clean way those two builds, espacially regarding GPU builds. Indeed
# bitpit can't build with nvcc (too many parsing errors from nvcc front 
# end), so bitpit is build with a regular CPU compiler (defaults to g++)
# and dyablo+kokkos with a compiler adapted to the target architecture :
# - g++ or clang++ or icpc for Kokkos/OpenMP
# - nvcc_wrapper Kokkos/CUDA
#
###################################################################
project (dyablo_superbuild LANGUAGES C CXX)


#
# default local cmake macro repository
#
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/core/cmake)

#
# Prevent from build in source tree
#
include(preventBuildInSource)

#
# dyablo options
#

option (DYABLO_USE_MPI "Activate / want MPI build" ON)
option (DYABLO_USE_VTK "Activate / want VTK build" OFF)
option (DYABLO_USE_DOUBLE "build with double precision" ON)
#option (USE_MOOD "build MOOD numerical schemes" OFF)
#option (USE_SDM "build Spectral Difference Method numerical schemes" OFF)
option (DYABLO_USE_HDF5 "build HDF5 input/output support" ON)
option (DYABLO_USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option (DYABLO_USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)
option (DYABLO_USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)

# Enforce rebuilding bitpit
option(FORCE_BITPIT_BUILD "Enforce rebuilding bitpit" OFF)

#
# documentation related options
#
option(BUILD_DOC  "Enable / disable documentation build" OFF)

# documentation type - the only valid values are : doxygen and mkdocs
set(DOC "doxygen" CACHE STRING "documentation type (doxygen or mkdocs)")

###################################################################
#
# this is where the superbuild starts
#
###################################################################

# dyablo and bitpit will for sure be built as external projects
include (ExternalProject)

##############################################################
# let's try first to detect bitpit
# this will only work if option -DBITPIT_DIR=/some/where
# was given on the command line
# the path should contain file BITPITConfig.cmake
##############################################################
find_package(BITPIT CONFIG QUIET)

# retrieve BITPIT_DIR as given on the command line
set(MY_BITPIT_DIR ${BITPIT_DIR})

# enforce a CPU build of BITPIT/PABLO
# anyway PABLO is a CPU library
# anyway PABLO is not at all in a shape to pass nvcc_wrapper compilation
# anyway we won't use PABLO internal structure in the core Kokkos
# computing functors
set(BITPIT_COMPILER  "g++" CACHE STRING "compiler used to build bitpit")

# list of dependencies to build before dyablo itself;
# currently the only dependency is bitpit external project;
# kokkos is still built inline
# DEPENDENCIES is initialized as empty, but in case BITPIT_FOUND
# is false, we add bitpit to DEPENDENCIES
set (DEPENDENCIES)

####################################################
#
# if BITPIT is not found, then build / install it
# then use our custom BITPIT_DIR to build dyablo
#
####################################################
if ( (NOT BITPIT_FOUND) OR FORCE_BITPIT_BUILD )

  list (APPEND DEPENDENCIES bitpit_external)

  #
  # build our custom BitPit version
  #
  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external)
  ExternalProject_Add (bitpit_external
    URL https://github.com/pkestene/bitpit/archive/bitpit-1.7.0-devel-dyablo-v0.2.tar.gz
    URL_MD5 5d7a8672b711addc01708ac483d74b5c
    UPDATE_COMMAND ""
    CMAKE_ARGS
      -DCMAKE_CXX_COMPILER=${BITPIT_COMPILER}
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DENABLE_MPI=ON
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
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
    )

  set (BITPIT_VERSION_SHORT 1.7)

  set(MY_BITPIT_DIR "${CMAKE_BINARY_DIR}/external/Install/bitpit_external/lib/cmake/bitpit-${BITPIT_VERSION_SHORT}")

endif()

#####################################
# now we build dyablo
#####################################

# at some point we should also build kokkos using ExternalProject, instead of including
# its build system into dyablo

#
# prepare cmake arguments list
#
set (DYABLO_CMAKE_ARGS)

# some top-level cmake option (super-build)
option(Kokkos_ENABLE_HWLOC  "enable HWLOC in Kokkos" ON)
option(Kokkos_ENABLE_OPENMP "enable Kokkos::OpenMP backend" ON)
option(Kokkos_ENABLE_CUDA   "enable Kokkos::Cuda backend" OFF)
option(Kokkos_ENABLE_CUDA_LAMBDA   "enable Kokkos::Cuda backend with LAMBDA" ON)
option(Kokkos_ENABLE_CUDA_CONSTEXPR   "enable Kokkos::Cuda backend with LAMBDA" OFF)
option(Kokkos_ENABLE_CUDA_UVM   "enable Kokkos::Cuda backend with LAMBDA" OFF)

option(Kokkos_ENABLE_DEBUG  "build Kokkos in debug mode" OFF)
option(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK  "build Kokkos in debug mode" OFF)

# if CUDA is enable, you need to be careful about the version of boost
option(DYABLO_ENABLE_UNIT_TESTING "Enable unit testing" OFF)

# only usefull when building for Kokkos::Cuda backend 
set(Kokkos_ARCH  "" CACHE STRING "Kokkos arch (KEPLER37, PASCAL60, ...)")

# default arguments - common to all platform
list (APPEND DYABLO_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DKokkos_ENABLE_HWLOC=${Kokkos_ENABLE_HWLOC}
  -DKokkos_ENABLE_OPENMP=${Kokkos_ENABLE_OPENMP}
  -DKokkos_ENABLE_DEBUG=${Kokkos_ENABLE_DEBUG}
  -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=${Kokkos_ENABLE_DEBUG_BOUNDS_CHECK}
  -DDYABLO_USE_HDF5=${DYABLO_USE_HDF5}
  -DDYABLO_USE_PNETCDF=${DYABLO_USE_PNETCDF}
  -DDYABLO_USE_VTK=${DYABLO_USE_VTK}
  -DDYABLO_USE_MPI=${DYABLO_USE_MPI}
  -DDYABLO_USE_MPI_CUDA_AWARE_ENFORCED=${DYABLO_USE_MPI_CUDA_AWARE_ENFORCED}
  -DDYABLO_USE_DOUBLE=${DYABLO_USE_DOUBLE}
  -DDYABLO_USE_FPE_DEBUG=${DYABLO_USE_FPE_DEBUG}
  -DBUILD_DOC=${BUILD_DOC}
  -DDOC=${DOC}
  -DDYABLO_ENABLE_UNIT_TESTING=${DYABLO_ENABLE_UNIT_TESTING}
  -DBITPIT_DIR=${MY_BITPIT_DIR}
  )

# CUDA specific argument
if (Kokkos_ENABLE_CUDA)

  list (APPEND DYABLO_CMAKE_ARGS
    -DKokkos_ENABLE_CUDA=${Kokkos_ENABLE_CUDA}
    -DKokkos_ENABLE_CUDA_LAMBDA=${Kokkos_ENABLE_CUDA_LAMBDA}
    -DKokkos_ENABLE_CUDA_CONSTEXPR=${Kokkos_ENABLE_CUDA_CONSTEXPR}
    -DKokkos_ENABLE_CUDA_UVM=${Kokkos_ENABLE_CUDA_UVM}
    -DKokkos_ARCH_${Kokkos_ARCH}=ON
    )

endif()

#
# build dyablo
#
ExternalProject_Add (dyablo_external
  DEPENDS ${DEPENDENCIES}
  BUILD_ALWAYS TRUE
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/core
  CMAKE_ARGS ${DYABLO_CMAKE_ARGS}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dyablo)
