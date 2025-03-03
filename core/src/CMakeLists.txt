set( core_src
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh_pablo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh_hashmap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh_hashmap_new.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/MapUserData_mean.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hydro/HydroUpdate_hancock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hydro/HydroUpdate_hancock_oneneighbor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hydro/HydroUpdate_euler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hydro/HydroUpdate_RK2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_point_mass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_constant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_cg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_hydro.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_cosmology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_particle_velocity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_parabolic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_second_derivative_error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_pseudo_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_mass.cpp    
    ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_downflows.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/particles/ParticleUpdate_tracers_move.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/particles/ParticleUpdate_NGP_move.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/particles/ParticleUpdate_NGP_density.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/particles/ParticleUpdate_CIC_move.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/particles/ParticleUpdate_CIC_density.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parabolic/ParabolicUpdate_explicit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cooling/CoolingUpdate_FF.cpp
)

set( ic_src
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_blast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_implode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_riemann2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_kelvin_helmholtz.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_rayleigh_taylor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_orszag_tang.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_MHD_blast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_MHD_rotor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_simple_particles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_sod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_particle_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_grafic_fields.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/wholesun/InitialConditions_C91.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/wholesun/InitialConditions_tri_layer.cpp
)
     
set( legacy_src
    ${CMAKE_CURRENT_SOURCE_DIR}/legacy/CopyGhostBlockCellData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/legacy/MarkOctantsHydroFunctor.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/legacy/io_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hydro/HydroUpdate_legacy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_legacy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/refine_condition/RefineCondition_legacy.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/legacy/MapUserData_legacy.cpp
)

set( mpi_src
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/ViewCommunicator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/GhostCommunicator_partial_blocks.cpp
)

set( hdf5_src
    ${CMAKE_CURRENT_SOURCE_DIR}/io/IOManager_hdf5.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io/IOManager_checkpoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_restart.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_tiled_restart.cpp
)

add_library( dyablo_core ${core_src} ${ic_src} ${legacy_src} )
target_include_directories( dyablo_core PUBLIC . )
target_link_libraries( dyablo_core PUBLIC
    Kokkos::kokkos
    BITPIT::BITPIT
)

if (DYABLO_USE_MPI)
    target_compile_definitions(dyablo_core PUBLIC BITPIT_ENABLE_MPI)
    target_link_libraries( dyablo_core PUBLIC mpiUtils)
    target_sources( dyablo_core PRIVATE ${mpi_src} )
endif (DYABLO_USE_MPI)

if (DYABLO_USE_HDF5)
    #target_link_libraries( dyablo_core PUBLIC HDF5::HDF5 )
    #TODO : properly use all flags with target HDF5::HDF5 (including -D_FORTIFY_SOURCE=2)
    target_link_libraries( dyablo_core PUBLIC ${HDF5_LIBRARIES} )
    target_sources( dyablo_core PRIVATE ${hdf5_src} )
endif (DYABLO_USE_HDF5)

# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)
