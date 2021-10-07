#######################
### MISC ##############
#######################
add_library(misc "")

target_sources(misc
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
)

target_include_directories(misc
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils/misc
  )
target_link_libraries( misc PRIVATE HostOnly )

add_library(dyablo::misc ALIAS misc)
