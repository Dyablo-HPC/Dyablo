include (ExternalProject)
  
# list of dependencies to dyablo; currently only bitpit external project
set (DEPENDENCIES)
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

set (EXTRA_CMAKE_ARGS)
list (APPEND EXTRA_CMAKE_ARGS
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DKokkos_ENABLE_HWLOC=ON
  -DKokkos_ENABLE_OPENMP=ON
  -DUSE_HDF5=ON
  -DUSE_MPI=ON
  -DBITPIT_DIR=${CMAKE_BINARY_DIR}/external/Install/bitpit_external/lib/cmake/bitpit-${BITPIT_VERSION_SHORT}
  )

#
# build dyablo
#
ExternalProject_Add (dyablo_external
  DEPENDS ${DEPENDENCIES}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}
  CMAKE_ARGS -DUSE_SUPERBUILD=OFF ${EXTRA_CMAKE_ARGS}
  INSTALL_COMMAND ""
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dyablo)
