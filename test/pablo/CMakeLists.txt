add_executable(PABLO_test1 PABLO_test1.cpp)
target_include_directories(PABLO_test1
  PUBLIC
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/common
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/containers
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/IO
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/operators
  ${CMAKE_SOURCE_DIR}/external/bitpit/src/PABLO)
target_link_libraries(PABLO_test1 ${BITPIT_LIBRARY})
