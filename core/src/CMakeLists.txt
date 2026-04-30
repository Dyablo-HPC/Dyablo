set( core_src
    ${CMAKE_CURRENT_SOURCE_DIR}/DyabloSession.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/AMRmesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/LightOctree_hashmap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/user_data/UserData_fields.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/user_data/UserData_particles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/foreach_cell/Binned_iOcts.cpp
)

set( plugins_src
    ${CMAKE_CURRENT_SOURCE_DIR}/plugins_lib.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/MapUserData_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/MapUserData_mean.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amr/MapUserData_linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/scheme/HyperbolicUpdate_hancock_oneneighbor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/HydroUpdate_gravity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/HydroUpdate_euler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/HydroUpdate_euler_nopatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/HydroUpdate_hancock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/HydroUpdate_RK2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/GLMMHDUpdate_euler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/GLMMHDUpdate_hancock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/GLMMHDUpdate_RK2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hyperbolic/RadUpdate_euler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_point_mass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_constant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gravity/GravitySolver_cg.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_hyperbolic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_hydro_samurai.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_cosmology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_particle_velocity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_parabolic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_rad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_fixed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute_dt/Compute_dt_BZ.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/source_terms/SourceUpdate_Cooling_FF.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source_terms/SourceUpdate_GLM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source_terms/SourceUpdate_BZ.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io/derived_fields/DerivedFields_divB.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source_terms/SourceUpdate_Photons_Beam.cpp
)

set( ic_src
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_uniform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/hydro/InitialConditions_blast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/hydro/InitialConditions_implode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/hydro/InitialConditions_riemann2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/hydro/InitialConditions_kelvin_helmholtz.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/hydro/InitialConditions_rayleigh_taylor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/hydro/InitialConditions_sod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/hydro/InitialConditions_double_mach.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/hydro/InitialConditions_heat_conduction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/mhd/InitialConditions_orszag_tang.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/mhd/InitialConditions_MHD_blast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/mhd/InitialConditions_MHD_rotor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/mhd/InitialConditions_MHD_rayleigh_taylor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/mhd/InitialConditions_MHD_Brio_Wu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/mhd/InitialConditions_MHD_isodensity_vortex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/mhd/InitialConditions_MHD_cloud_shock_interaction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_simple_particles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_particle_grid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_particles_uniform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_grafic_fields.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/wholesun/InitialConditions_C91.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/wholesun/InitialConditions_tri_layer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_zeldovitch_pancake.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_zeldovitch_particles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_BZ.cpp
)

set( mpi_src
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/ViewCommunicator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/GhostCommunicator_partial_blocks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/GhostCommunicator_full_blocks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mpi/GhostCommunicator_Subset_levels.cpp
)

set( hdf5_src
    ${CMAKE_CURRENT_SOURCE_DIR}/io/IOManager_hdf5.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io/IOManager_checkpoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/io/IOManager_diagnostics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_restart.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init/InitialConditions_tiled_restart.cpp
)
# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)

add_library( dyablo_core ${core_src})
target_include_directories( dyablo_core PUBLIC . )
target_link_libraries( dyablo_core PUBLIC
    config
    dyablo_units
    Kokkos::kokkos
    monitoring
    dl
)

if (DYABLO_USE_MPI)
    target_link_libraries( dyablo_core PUBLIC mpiUtils)
    target_sources( dyablo_core PRIVATE ${mpi_src} )
endif (DYABLO_USE_MPI)


add_library( dyablo_plugins ${ic_src} ${plugins_src})
target_link_libraries( dyablo_plugins PRIVATE dyablo_core )
if (DYABLO_USE_HDF5)
    #target_link_libraries( dyablo_core PUBLIC HDF5::HDF5 )
    #TODO : properly use all flags with target HDF5::HDF5 (including -D_FORTIFY_SOURCE=2)
    target_link_libraries( dyablo_plugins PUBLIC ${HDF5_LIBRARIES} )
    target_sources( dyablo_plugins PRIVATE ${hdf5_src} )
endif (DYABLO_USE_HDF5)


