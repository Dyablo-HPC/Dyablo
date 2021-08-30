add_library(mpiUtils "")

target_sources (mpiUtils
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/GlobalMpiSession.cpp
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/GlobalMpiSession.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.h
  )
#Set target properties
target_include_directories(mpiUtils
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
)
target_link_libraries (mpiUtils PUBLIC MPI::MPI_CXX)
target_link_libraries (mpiUtils PRIVATE HostOnly)

#Add an alias so that library can be used inside the build tree
add_library(dyablo::utils::mpi ALIAS mpiUtils)
