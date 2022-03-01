set( core_src
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh_hashmap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hydro/HydroUpdate_generic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hydro/HydroUpdate_muscl_oneneighbor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_constant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_cg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_generic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_second_derivative_error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_pseudo_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldManager.cpp
)

set( ic_src
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_blast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_implode.cpp
)
     
set( legacy_src
    ${CMAKE_CURRENT_SOURCE_DIR}/legacy/MapUserData.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/legacy/CopyGhostBlockCellData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/legacy/MarkOctantsHydroFunctor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/legacy/io_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hydro/HydroUpdate_legacy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_legacy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_legacy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io/IOManager_hdf5.cpp
)

set( mpi_src
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/GhostCommunicator_kokkos.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/GhostCommunicator_pablo.cpp
)

set( hdf5_src
    ${CMAKE_CURRENT_SOURCE_DIR}/legacy/HDF5_IO.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io/IOManager_hdf5.cpp
)

add_library( dyablo_core ${core_src} ${ic_src} ${legacy_src} )
target_include_directories( dyablo_core PUBLIC . )
target_link_libraries( dyablo_core PUBLIC
    Kokkos::kokkos
    BITPIT::BITPIT
)

if (DYABLO_USE_MPI)
    target_compile_definitions(dyablo_core PUBLIC BITPIT_ENABLE_MPI)
    target_sources( dyablo_core PRIVATE ${mpi_src} )
endif (DYABLO_USE_MPI)

if (DYABLO_USE_HDF5)
    #target_link_libraries( dyablo_core PUBLIC HDF5::HDF5 )
    #TODO : properly use all flags with target HDF5::HDF5 (including -D_FORTIFY_SOURCE=2)
    target_link_libraries( dyablo_core PUBLIC ${HDF5_LIBRARIES} )
    target_link_libraries( dyablo_core PUBLIC io )
    target_sources( dyablo_core PRIVATE ${hdf5_src} )
endif (DYABLO_USE_HDF5)

# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)