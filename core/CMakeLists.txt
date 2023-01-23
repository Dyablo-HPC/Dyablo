# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.16)

project(dyablo C CXX)

# default local cmake macro repository
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# Prevent from build in source tree
include(preventBuildInSource)
# Write a small header with build data, git version, etc...
include(write_version)
# Init build type: Release, Debug, ...
include(initBuildType)
# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# C++14 is for Kokkos
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# dyablo options from Superbuild
option (DYABLO_ENABLE_UNIT_TESTING "Enable unit testing" OFF)
option (DYABLO_USE_MPI "Activate / want MPI build" ON)
option (DYABLO_USE_HDF5 "build HDF5 input/output support" ON)
option (DYABLO_USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)
option (DYABLO_USE_INTERNAL_KOKKOS "Use internal Kokkos from external/kokkos" ON)
option (Kokkos_ENABLE_CUDA   "enable Kokkos::Cuda backend" OFF)
option (Kokkos_ENABLE_SYCL   "enable Kokkos::SyCL backend" OFF)
option (DYABLO_USE_PABLO "use AMRmesh_pablo for AMR" OFF)
set    (BITPIT_DIR "BITPIT package directory" CACHE PATH ""  )
set    (Kokkos_DIR "Kokkos installation directory or internal source" CACHE PATH ""  )


# Options not forwarded from superbuild
option (DYABLO_BUILD_EXPERIMENTS "Build optional experiments from core/test" OFF)
option (DYABLO_BUILD_DOC  "Enable / disable documentation build" OFF)
set    (DYABLO_DOC "doxygen" CACHE STRING "documentation type (doxygen or mkdocs)")
option (DYABLO_USE_DOUBLE "build with double precision" ON)
option (DYABLO_USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)
option (Kokkos_ENABLE_OPENMP "enable Kokkos::OpenMP backend" ON)
set (Kokkos_ENABLE_CUDA_LAMBDA ${Kokkos_ENABLE_CUDA} CACHE BOOL "Enable use of cuda lambdas" FORCE)
set (Kokkos_ENABLE_CUDA_CONSTEXPR ${Kokkos_ENABLE_CUDA} CACHE BOOL "Enable use of cuda constexpr" FORCE)

# Use CMAKE_DL_LIBS to find the dl library
set( LIBDL_LIBRARY CACHE STRING "LIBDL library" "${CMAKE_DL_LIBS}" )

#####################################################################
# BitPit
#####################################################################
if( DYABLO_USE_PABLO )
  find_package(BITPIT REQUIRED)
  # create imported target for bitpit library
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
else()
  add_library( BITPIT::BITPIT INTERFACE IMPORTED GLOBAL )
endif()
#####################################################################
# Kokkos
#####################################################################
if( DYABLO_USE_INTERNAL_KOKKOS )
  FetchContent_Declare( kokkos_external
    SOURCE_DIR ${Kokkos_DIR}
  )
  # Import kokkos targets
  FetchContent_MakeAvailable(kokkos_external)
else( DYABLO_USE_INTERNAL_KOKKOS )
  find_package( Kokkos 3.5 REQUIRED )
  kokkos_check( DEVICES "OpenMP" )
  if(Kokkos_ENABLE_CUDA)
    # kokkos_check is defined in KokkosConfigCommon.cmake
    kokkos_check( DEVICES "Cuda" )
    kokkos_check( OPTIONS CUDA_LAMBDA)
  endif()
endif()

# Check compiler version
message( "Detected kokkos compiler : ${KOKKOS_CXX_COMPILER_ID} ${KOKKOS_CXX_COMPILER_VERSION}" )
set( NVCC_FIRST_VERSION 11.2 )
if(   KOKKOS_CXX_COMPILER_ID STREQUAL "NVIDIA" AND KOKKOS_CXX_COMPILER_VERSION LESS NVCC_FIRST_VERSION )
  set( DYABLO_ENABLE_DEPRECATED_NVCC "Use deprecated NVCC" CACHE STRING "I want right results"  )
  if( DYABLO_ENABLE_DEPRECATED_NVCC STREQUAL "I want wrong results" )
    message( WARNING "nvcc version ${KOKKOS_CXX_COMPILER_VERSION} (< ${NVCC_FIRST_VERSION}) is deprecated but you chose to use it anyway. That may cause wrong numerical results." )
  else()
    message( FATAL_ERROR "nvcc version ${KOKKOS_CXX_COMPILER_VERSION} (< ${NVCC_FIRST_VERSION}) is deprecated because it may cause wrong numerical results. If you want to use it anyway, set DYABLO_ENABLE_DEPRECATED_NVCC to \"I want wrong results\" " )
  endif()
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
    add_compile_options(-DDYABLO_USE_HDF5)
    if (HDF5_IS_PARALLEL)
      add_compile_options(-DDYABLO_USE_HDF5_PARALLEL)
    endif()
  endif(HDF5_FOUND)
endif(DYABLO_USE_HDF5)

#####################################################################
# Backward-cpp
#####################################################################
FetchContent_Declare( backward-cpp_external
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/../external/backward-cpp
)
# Import kokkos targets
FetchContent_MakeAvailable(backward-cpp_external)

# common flags
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
if (DYABLO_USE_PABLO)
  add_compile_options(-DDYABLO_USE_PABLO)
endif()

##
## Using flags -Wextra, it's to strong for Kokkos, too many warnings
## But -Wall is really a minimum
##
#add_definitions(-Wall -Wextra -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wcast-qual -Wunused -Woverloaded-virtual  -pedantic )
#add_definitions( -Wall -Wextra )
add_definitions( -Wall )

if( DYABLO_BUILD_EXPERIMENTS )
add_subdirectory(test)
else()
add_subdirectory(test/solver)
endif()

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
  message("  Kokkos_ENABLE_SYCL=${Kokkos_ENABLE_SYCL}")
  message("  Kokkos_ARCH=${Kokkos_ARCH}")
endif(Kokkos_FOUND)
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

