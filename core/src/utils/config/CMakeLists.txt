add_library(config "")

target_sources(config
  PUBLIC
  #${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.h
  )

target_include_directories(config
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/inih ) 

target_link_libraries(config PUBLIC dyablo::utils::mpi)

#Add an alias so that library can be used inside the build tree
add_library(dyablo::config ALIAS config)

