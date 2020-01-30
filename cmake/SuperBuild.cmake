##############################################################
# let try first to detect bitpit
# this will only worked if option -DBITPIT_DIR=/some/where
# was given on the command line
# the path should contain file BITPITConfig.cmake
##############################################################
find_package(BITPIT CONFIG QUIET)

# retrieve BITPIT_DIR as given on the command line
set(MY_BITPIT_DIR ${BITPIT_DIR})

# enforce a CPU build of BITPIT/PABLO
# anyway PABLO is a CPU library
# anyway PABLO is not a all in a shape to pass nvcc_wrapper compilation
# anyway we won't use PABLO internal structure in tge core Kokkos
# computing functors
set(BITPIT_COMPILER  "g++" CACHE STRING "compiler used to build bitpit")

# Enforce rebuilding bitpit
option(FORCE_BITPIT_BUILD "Enforce rebuilding bitpit" OFF)

# list of dependencies to dyablo; currently only bitpit external project
# initialized as empty, but in case BITPIT_FOUND is false, we add bitpit
# to DEPENDENCIES
set (DEPENDENCIES)

# dyablo will for sure be built as an external project
include (ExternalProject)  

####################################################
#
# if BITPIT is not found, then build / install it
# then use our custom BITPIT_DIR to build dyablo
#
####################################################
if ( (NOT BITPIT_FOUND) OR FORCE_BITPIT_BUILD)

  list (APPEND DEPENDENCIES bitpit_external)

  #
  # build our custom BitPit version
  #
  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external)
  ExternalProject_Add (bitpit_external
    URL https://github.com/pkestene/bitpit/archive/bitpit-1.7.0-devel-dyablo-v0.1.tar.gz
    URL_MD5 3c2efb80dbf00cec22c813e15119444f
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

#
# prepare cmake arguments list
#
set (DYABLO_CMAKE_ARGS)

option(Kokkos_ENABLE_HWLOC  "enable HWLOC in Kokkos" ON)
option(Kokkos_ENABLE_OPENMP "enable Kokkos::OpenMP backend" ON)
option(Kokkos_ENABLE_CUDA   "enable Kokkos::Cuda backend" OFF)

# only usefull when building for Kokkos::Cuda backend 
set(Kokkos_ARCH  "" CACHE STRING "Kokkos arch (KEPLER37, PASCAL60, ...)")

if (Kokkos_ENABLE_CUDA)

  list (APPEND DYABLO_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DKokkos_ENABLE_HWLOC=${Kokkos_ENABLE_HWLOC}
    -DKokkos_ENABLE_OPENMP=${Kokkos_ENABLE_OPENMP}
    -DKokkos_ENABLE_CUDA=${Kokkos_ENABLE_CUDA}
    -DKokkos_ENABLE_CUDA_LAMBDA=ON
    -DKokkos_ARCH_${Kokkos_ARCH}=ON
    -DUSE_HDF5=ON
    -DUSE_MPI=ON
    -DBITPIT_DIR=${MY_BITPIT_DIR}
    )

else()

  list (APPEND DYABLO_CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DKokkos_ENABLE_HWLOC=${Kokkos_ENABLE_HWLOC}
    -DKokkos_ENABLE_OPENMP=${Kokkos_ENABLE_OPENMP}
    -DUSE_HDF5=ON
    -DUSE_MPI=ON
    -DBITPIT_DIR=${MY_BITPIT_DIR}
    )

endif()

#
# build dyablo
#
ExternalProject_Add (dyablo_external
  DEPENDS ${DEPENDENCIES}
  BUILD_ALWAYS TRUE
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_ARGS -DUSE_SUPERBUILD=OFF ${DYABLO_CMAKE_ARGS}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dyablo)
