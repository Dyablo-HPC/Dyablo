# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.10)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
  message(STATUS "Setting policy CMP0076 to use new behavior")
  cmake_policy(SET CMP0076 NEW)
endif()

set(PROJECT_LANGUAGES ${PROJECT_LANGUAGES} C CXX)
project(dyablo
  LANGUAGES ${PROJECT_LANGUAGES})

#
# default local cmake macro repository
#
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#
# Prevent from build in source tree
#
include(preventBuildInSource)

#
# Write a small header with build data, git version, etc...
#
include(write_version)

#
# Init build type: Release, Debug, ...
#
include(initBuildType)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# dyablo options are defined in top level CMakeLists.txt

# C++14 is for Kokkos
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

#####################################################################
# External packages: MPI, ...
#####################################################################

#####################################################################
# BitPit
#####################################################################
find_package(BITPIT REQUIRED)
# create imported target for bitpit library
if (BITPIT_FOUND)
  if(NOT TARGET BITPIT::BITPIT)
    get_filename_component(LIB_EXT "${BITPIT_LIBRARY}" EXT)
    if(LIB_EXT STREQUAL ".a" OR LIB_EXT STREQUAL ".lib")
      set(LIB_TYPE STATIC)
    else()
      set(LIB_TYPE SHARED)
    endif()
    add_library(BITPIT::BITPIT ${LIB_TYPE} IMPORTED GLOBAL)
    set(_tmp_dep_libs "${BITPIT_LIBRARIES}")
    list(REMOVE_DUPLICATES _tmp_dep_libs)
    set_target_properties(BITPIT::BITPIT
      PROPERTIES
      IMPORTED_LOCATION "${BITPIT_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${BITPIT_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${_tmp_dep_libs}")
  endif()
endif(BITPIT_FOUND)

#####################################################################
# Kokkos
#####################################################################
find_package(Kokkos 3.1 REQUIRED)
if(Kokkos_ENABLE_CUDA)
  # kokkos_check is defined in KokkosConfigCommon.cmake
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

#####################################################################
# MPI
#####################################################################
find_package(MPI)
if (DYABLO_USE_MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})
    
    #set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
    if(Kokkos_ENABLE_CUDA)
      find_program(OMPI_INFO
        NAMES ompi_info
        PATHS ${MPI_CXX_LIBRARIES}/../bin)
    
      message( STATUS "MPI checking CUDA support..." )
      # Full command line to probe if cuda support in MPI implementation is enabled
      # ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
      if (OMPI_INFO)
        message( STATUS "  Found ompi_info" )
        execute_process(COMMAND ${OMPI_INFO}
          OUTPUT_VARIABLE _output)
        if ( (_output MATCHES "smcuda") )
          message( STATUS "  OpenMPI is CUDA Aware" )
          set(MPI_CUDA_AWARE_DETECTED ON)
        else()
          message( STATUS "  OpenMPI is NOT CUDA Aware" )
        endif()
      else()
	# Not using OpenMPI and not enforcing CUDA Aware flag -> Warning
	if (NOT DYABLO_USE_MPI_CUDA_AWARE_ENFORCED)
          message( WARNING "  Could not find ompi_info. MPI is supposed NOT CUDA Aware. (To override, set DYABLO_USE_MPI_CUDA_AWARE_ENFORCED)")
	endif()
      endif()
      if( DYABLO_USE_MPI_CUDA_AWARE_ENFORCED OR MPI_CUDA_AWARE_DETECTED )
        message(STATUS "MPI CUDA Aware : ON")
        add_compile_options(-DMPI_IS_CUDA_AWARE)
        if( NOT MPI_CUDA_AWARE_DETECTED )
          message( WARNING "Dyablo uses CUDA Aware MPI calls but could not determine if MPI supports CUDA !" )
        endif()
      else()
        message(STATUS "MPI CUDA Aware : OFF")
      endif()
    endif()
  else()
    message(WARNING "Not compiling with MPI. Suppress this warning with -DDYABLO_USE_MPI=OFF")
    set(DYABLO_USE_MPI OFF)
  endif()
endif()


#####################################################################
# VTK configuration tips, see
# /usr/lib/cmake/vtk-6.2/VTKConfig.cmake
# /usr/lib/cmake/vtk-6.2/UseVTK.cmake
#####################################################################
if (DYABLO_USE_VTK)
  # look for VTK only if requested; VTK macro might even be not present
  # on the target platform
  find_package(VTK)
  
  # the following add VTK to all targets
  # if(VTK_FOUND)
  #   include(${VTK_USE_FILE})
  # endif(VTK_FOUND)
  if (VTK_FOUND)
    message("***VTK FOUND ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
  else()
    message ("*** VTK NOT FOUND")
  endIF()
endif(DYABLO_USE_VTK)

#####################################################################
# HDF5
#####################################################################
# prefer using parallel HDF5 when build with mpi
if (DYABLO_USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif(DYABLO_USE_MPI)

if (DYABLO_USE_HDF5)
  find_package(HDF5)
  if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    set(my_hdf5_libs hdf5 hdf5_cpp)
    add_compile_options(-DDYABLO_USE_HDF5 -DH5_NO_DEPRECATED_SYMBOLS)
    if (HDF5_IS_PARALLEL)
      add_compile_options(-DDYABLO_USE_HDF5_PARALLEL)
    endif()
  endif(HDF5_FOUND)
endif(DYABLO_USE_HDF5)

#####################################################################
# PNETCDF
#####################################################################
if (DYABLO_USE_MPI)
  if (DYABLO_USE_PNETCDF)
    find_package(PNETCDF)
    if (PNETCDF_FOUND)
      add_compile_options(-DDYABLO_USE_PNETCDF)
      include_directories(${PNETCDF_INCLUDE_DIRS})
    endif(PNETCDF_FOUND)
  endif(DYABLO_USE_PNETCDF)
endif(DYABLO_USE_MPI)  

#
# common flags
#

if (Kokkos_ENABLE_CUDA)
  add_compile_options(-DCUDA)
endif()

if (DYABLO_USE_DOUBLE)
  add_compile_options(-DDYABLO_USE_DOUBLE)
endif()

if (DYABLO_USE_MPI)
  add_compile_options(-DDYABLO_USE_MPI)    
endif()

if (DYABLO_USE_FPE_DEBUG)
  add_compile_options(-DDYABLO_USE_FPE_DEBUG)
endif()

##
## Using flags -Wextra, it's to strong for Kokkos, too many warnings
## But -Wall is really a minimum
##
#add_definitions(-Wall -Wextra -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wcast-qual -Wunused -Woverloaded-virtual  -pedantic )
#add_definitions( -Wall -Wextra )
add_definitions( -Wall )

#
# sources
#
add_subdirectory(../external/backward-cpp external/backward-cpp)
#add_subdirectory(../external/kokkos external/kokkos)

# pass Kokkos include directories to our target application
#include_directories(${Kokkos_INCLUDE_DIRS_RET})

add_subdirectory(test)
add_subdirectory(src)

# Documentation type
if(DYABLO_BUILD_DOC)
  add_subdirectory(doc)
endif(DYABLO_BUILD_DOC)

# unit testing
if (DYABLO_ENABLE_UNIT_TESTING)
  enable_testing()
  add_subdirectory(unit_test)
endif()

##################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} " 
  "${CMAKE_CXX_COMPILER_VERSION} " 
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
if (DYABLO_USE_MPI)
  message("  MPI enabled")
else(DYABLO_USE_MPI)
  message("  MPI not enabled")
endif(DYABLO_USE_MPI)
if (Kokkos_FOUND)
  message("  Kokkos_Found true")
  message("  Kokkos_ENABLE_HWLOC=${Kokkos_ENABLE_HWLOC}")
  message("  Kokkos_ENABLE_OPENMP=${Kokkos_ENABLE_OPENMP}")
  message("  Kokkos_ENABLE_DEBUG=${Kokkos_ENABLE_DEBUG}")
  message("  Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=${Kokkos_ENABLE_DEBUG_BOUNDS_CHECK}")
  message("  Kokkos_ENABLE_CUDA=${Kokkos_ENABLE_CUDA}")
  message("  Kokkos_ENABLE_CUDA_LAMBDA=${Kokkos_ENABLE_CUDA_LAMBDA}")
  message("  Kokkos_ENABLE_CUDA_CONSTEXPR=${Kokkos_ENABLE_CUDA_CONSTEXPR}")
  message("  Kokkos_ENABLE_CUDA_UVM=${Kokkos_ENABLE_CUDA_UVM}")
  message("  Kokkos_ARCH=${Kokkos_ARCH}")
endif(Kokkos_FOUND)

if (Kokkos_ENABLE_CUDA)
  message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
  message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
  message("  CUDA Compile flags : ${CUDA_NVCC_FLAGS}")
else(Kokkos_ENABLE_CUDA)
  message("  CUDA not enabled")
endif(Kokkos_ENABLE_CUDA)
if (DYABLO_USE_HDF5)
  if (HDF5_FOUND)
    message("  HDF5 found version    : ${HDF5_VERSION}")
    message("  HDF5 definitions      : ${HDF5_DEFINITIONS}")
    message("  HDF5 parallel         : ${HDF5_IS_PARALLEL}")
    message("  HDF5 includes dirs    : ${HDF5_INCLUDE_DIRS}")
    message("  HDF5 libraries        : ${HDF5_LIBRARIES}")
  endif(HDF5_FOUND)
endif(DYABLO_USE_HDF5)
message("")
if (DYABLO_BUILD_DOC)
  message("   Building doc of type ${DYABLO_DOC}")
endif (DYABLO_BUILD_DOC)
message("")

