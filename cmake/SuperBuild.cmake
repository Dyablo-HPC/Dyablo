##############################################################
# let try first to detect bitpit
# this will only worked if option -DBITPIT_DIR=/some/where
# was given on the command line
# the path should contain file BITPITConfig.cmake
##############################################################
find_package(BITPIT CONFIG QUIET)

# retrieve BITPIT_DIR as given on the command line
set(MY_BITPIT_DIR ${BITPIT_DIR})

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
message("KK bitpit_found : ${BITPIT_FOUND}")
if (NOT BITPIT_FOUND)

  message("KK bitpit NOT found")

  list (APPEND DEPENDENCIES bitpit_external)

  #
  # build our custom BitPit version
  #
  set_property(DIRECTORY PROPERTY EP_BASE ${CMAKE_BINARY_DIR}/external)
  ExternalProject_Add (bitpit_external
    URL https://github.com/pkestene/bitpit/archive/bitpit-1.7.0-devel-dyablo.tar.gz
    URL_MD5 496d20de8966d5dd03756e18c2351d41
    UPDATE_COMMAND ""
    CMAKE_ARGS
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
list (APPEND DYABLO_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DKokkos_ENABLE_HWLOC=ON
  -DKokkos_ENABLE_OPENMP=ON
  -DUSE_HDF5=ON
  -DUSE_MPI=ON
  -DBITPIT_DIR=${MY_BITPIT_DIR}
  )

#
# build dyablo
#
ExternalProject_Add (dyablo_external
  DEPENDS ${DEPENDENCIES}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_ARGS -DUSE_SUPERBUILD=OFF ${DYABLO_CMAKE_ARGS}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dyablo)
