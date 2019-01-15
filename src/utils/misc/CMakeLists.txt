#######################
### MISC ##############
#######################
set (misc_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
)
add_library(misc
  ${misc_SRCS}
  )
target_include_directories(misc
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils/misc
  )
