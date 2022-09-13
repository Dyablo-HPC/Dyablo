set (io_SRCS
  #${CMAKE_CURRENT_SOURCE_DIR}/io_utils.cpp
  #${CMAKE_CURRENT_SOURCE_DIR}/SimpleVTKIO.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FileHandler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FileHandler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FileHandlerVtk.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FileHandlerVtk.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKWriter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKWriter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK_shared.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IO_VTK_shared.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libb64/libb64.h
  ${CMAKE_CURRENT_SOURCE_DIR}/libb64/cencode.c
  )

add_library(io
  ${io_SRCS}
  )

target_include_directories(io
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  )
target_link_libraries(io
  config
  misc
  Kokkos::kokkos
  )
