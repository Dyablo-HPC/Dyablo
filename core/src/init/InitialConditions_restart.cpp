#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "userdata_utils.h"
#include "UserData.h"

#include <hdf5.h>
#include <hdf5_hl.h>

namespace dyablo{

namespace{

template<typename T>
hid_t get_hdf5_type()
{
  static_assert( !std::is_same_v<T,T>, "get_hdf5_type not defined for this type" );
  return 0;
}
template<> [[maybe_unused]] hid_t get_hdf5_type<double>()    { return H5T_NATIVE_DOUBLE; }
template<> [[maybe_unused]] hid_t get_hdf5_type<uint32_t>()  { return H5T_NATIVE_UINT32; }
template<> [[maybe_unused]] hid_t get_hdf5_type<int32_t>()   { return H5T_NATIVE_INT32; }


class restart_file{
private:
  hid_t m_hdf5_file;

public:
  restart_file( const restart_file& ) = delete;

  restart_file(const std::string& filename)
  {
    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
#ifdef DYABLO_USE_MPI
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
#endif
    m_hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist);
    H5Pclose(plist);
  }

  std::vector<std::string> list_fields( std::string hdf5_path )
  {
    std::vector<std::string> res;
    auto op_func = [](hid_t loc_id, const char *name, const H5L_info_t *info, void *operator_data)
    {
      std::vector<std::string>* res = static_cast<std::vector<std::string>*> (operator_data);
      res->push_back(name);
      return 0;
    };

    htri_t group_exists = H5Lexists(m_hdf5_file, hdf5_path.c_str(), H5P_DEFAULT);
    if( group_exists <= 0 ) // Group Does not exist : empty list
      return {};

    hid_t group_id = m_hdf5_file;
    if( hdf5_path!="" )
        group_id = H5Gopen(m_hdf5_file, hdf5_path.c_str(), H5P_DEFAULT);
    H5Literate(group_id, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, op_func, &res);
    if(group_id != m_hdf5_file)
      H5Gclose( group_id );
    return res;
  }

  std::vector<hsize_t> get_field_layout( std::string hdf5_path )
  {
    hid_t dataset_properties = H5Pcreate(H5P_DATASET_ACCESS);
    hid_t dataset = H5Dopen2( m_hdf5_file, hdf5_path.c_str(), dataset_properties );
    hid_t filespace = H5Dget_space( dataset );
    int rank = H5Sget_simple_extent_ndims(filespace);
    std::vector<hsize_t> dims(rank);
    H5Sget_simple_extent_dims( filespace, dims.data(), nullptr );
    H5Dclose(dataset);
    H5Sclose(filespace);
    H5Pclose(dataset_properties);

    return dims;
  }

  template< typename T, int iOct_pos=T::rank-1  >
  void read_view_init( const std::string& name, T& view )
  {
    hid_t hdf5_type = get_hdf5_type<typename T::value_type>();
    constexpr hid_t rank = (hid_t)T::rank;

    hid_t dataset_properties = H5Pcreate(H5P_DATASET_ACCESS);
    hid_t dataset = H5Dopen2( m_hdf5_file, name.c_str(), dataset_properties );
    hid_t filespace = H5Dget_space( dataset );

    DYABLO_ASSERT_HOST_RELEASE( rank == H5Sget_simple_extent_ndims(filespace), "hdf5 error : dataset rank doesn't match view rank" );
    hsize_t dims[rank], maxdims[rank];
    H5Sget_simple_extent_dims( filespace, dims, maxdims );
    
    Kokkos::LayoutLeft layout_file = view.layout();
    for(int i=0; i<rank; i++)
        layout_file.dimension[rank-1-i] = dims[i];
    T view_file(name, layout_file);
    {
      hid_t read_properties = H5Pcreate(H5P_DATASET_XFER);
      #ifdef HDF5_IS_CUDA_AWARE      
        H5Dread( dataset, hdf5_type, filespace, filespace, read_properties, view_file.data() );      
      #else
        auto view_file_host = Kokkos::create_mirror_view( view_file );
        H5Dread( dataset, hdf5_type, filespace, filespace, read_properties, view_file_host.data() );
        Kokkos::deep_copy( view_file, view_file_host );
      #endif
      H5Pclose(read_properties);
    }
    userdata_utils::transpose_from_right_iOct<iOct_pos>(view_file, view);

    H5Dclose(dataset);
    H5Sclose(filespace);
    H5Pclose(dataset_properties);
  }

  // iOct should be outermost index
  template< typename T >
  void read_view( const std::string& name, const T& view )
  {
    hid_t hdf5_type = get_hdf5_type<typename T::value_type>();
    constexpr hid_t rank = (hid_t)T::rank;
    hsize_t local_dims[rank];
    hsize_t global_dims[rank];
    hsize_t local_start[rank];
    for( int i=1; i<rank; i++ )
    {
       local_dims[i] = view.extent(rank-1-i);
       global_dims[i] = view.extent(rank-1-i);
       local_start[i] = 0;
    }
    hid_t local_oct_count = view.extent(rank-1);
    hid_t global_oct_count;
    MPI_Allreduce( &local_oct_count, &global_oct_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD );
    // Perform exclusive prefix sum
    hid_t local_oct_start;
    MPI_Scan( &local_oct_count, &local_oct_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD );
    local_oct_start -= local_oct_count;
    local_dims[0] = local_oct_count;
    global_dims[0] = global_oct_count;
    local_start[0] = local_oct_start;

    hid_t filespace = H5Screate_simple( rank, global_dims, nullptr );
    hid_t  memspace = H5Screate_simple( rank,  local_dims, nullptr );

    // set some properties
    hid_t dataset_properties = H5Pcreate(H5P_DATASET_ACCESS);    

    hid_t dataset = H5Dopen2( m_hdf5_file, name.c_str(), dataset_properties );
    H5Sselect_hyperslab( filespace, H5S_SELECT_SET, local_start, nullptr, local_dims, nullptr );
    
    {
      hid_t read_properties = H5Pcreate(H5P_DATASET_XFER);
      #ifdef DYABLO_USE_MPI
        H5Pset_dxpl_mpio(read_properties, H5FD_MPIO_COLLECTIVE);
      #endif
      #ifdef HDF5_IS_CUDA_AWARE      
        H5Dread( dataset, hdf5_type, filespace, filespace, read_properties, view.data() );      
      #else
        auto view_host = Kokkos::create_mirror_view( view );
        H5Dread( dataset, hdf5_type, memspace, filespace, read_properties, view_host.data() );
        Kokkos::deep_copy( view, view_host );
      #endif
      H5Pclose(read_properties);
    }

    H5Dclose(dataset);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Pclose(dataset_properties);
  }

  ~restart_file()
  {
    if (m_hdf5_file) {
      H5Fclose(m_hdf5_file);
      m_hdf5_file = 0;
    }
  }
};

} // namspace


class InitialConditions_restart : public InitialConditions{
    struct Data{
        ForeachCell& foreach_cell;
        
        int level_min, level_max;
    } data;
    std::string filename;
    ForeachParticle foreach_particle;
public:
  InitialConditions_restart(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : data({
      foreach_cell,
      foreach_cell.get_amr_mesh().get_level_min(),
      foreach_cell.get_amr_mesh().get_level_max()
    }),
    filename( configMap.getValue<std::string>( "restart", "filename", "restart.h5" ) ),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap)
  {}

  void init( UserData& U )
  {
    ForeachCell& foreach_cell = data.foreach_cell;
    AMRmesh& pmesh     = foreach_cell.get_amr_mesh();

    uint32_t ndim = pmesh.getDim();
    int level_min = data.level_min;
    int level_max = data.level_max;

    restart_file restart_file(filename);

    // Initialize AMR tree
    {
        LightOctree_storage<>::oct_data_t lmesh_data;
        restart_file.read_view_init<decltype(lmesh_data),0>( "Octree", lmesh_data );

        uint32_t nbCells_local = lmesh_data.extent(0);
        uint32_t nbCells_ghost = 0;

        // TODO : find a better way to retrieve coarse_grid_size
        LightOctree_storage<> lmesh_storage( ndim, nbCells_local, nbCells_ghost, level_min, pmesh.get_coarse_grid_size() ); 

        Kokkos::deep_copy( lmesh_storage.getLocalSubview(), lmesh_data );

        // Read mesh from restart file
        Kokkos::Array<bool,3> periodic{false, false, false};
        LightOctree_hashmap input_lmesh( std::move(lmesh_storage), level_min, level_max, periodic );

        std::cout << "Restart mesh : " << input_lmesh.getNumOctants() << " octs." << std::endl;

        // Refine until level_max using analytical markers
        for (uint8_t level=level_min; level<level_max; ++level)
        {
            const LightOctree& lmesh = pmesh.getLightOctree();
            uint32_t nbOcts = lmesh.getNumOctants();

            Kokkos::View<bool*> markers( "InitialConditions_restart::markers", nbOcts );

            // TODO parallelize with Kokkos
            Kokkos::parallel_for( "InitialConditions_restart::mark_octs", nbOcts,
              KOKKOS_LAMBDA( uint32_t iOct_new )
            {
                LightOctree::pos_t pos = lmesh.getCenter({iOct_new, false});
                auto oct_size = lmesh.getSize({iOct_new, false});
                pos[IX] += oct_size[IX]*0.01; // Add epsilon to avoid hitting boundary
                pos[IY] += oct_size[IY]*0.01;
                if(ndim==3)
                  pos[IZ] += oct_size[IZ]*0.01;

                LightOctree::OctantIndex iOct_input = input_lmesh.getiOctFromPos( pos );
                int level_input = input_lmesh.getLevel(iOct_input);

                markers(iOct_new) = ( level_input > level );
            });

            // Copy computed markers to pmesh
            auto markers_host = Kokkos::create_mirror_view(markers);
            Kokkos::deep_copy( markers_host, markers );
            Kokkos::parallel_for( "InitialConditions_restart::copy_octants",
                Kokkos::RangePolicy<Kokkos::OpenMP>(Kokkos::OpenMP(), 0, nbOcts),
                [&](uint32_t iOct)
            {
                if( markers_host(iOct) )
                    pmesh.setMarker(iOct, 1);
            });

            // Refine the mesh according to markers
            pmesh.adapt();        
        }
    }

    // TODO distribute mesh initialization
    // Load balance once at the end
    pmesh.loadBalance();

    // Reallocate and fill U
    {
      auto fields = restart_file.list_fields("fields/");
      
      for( std::string field : fields )
      {
        U.new_fields({field});
        ForeachCell::CellArray_global_ghosted field_view = foreach_cell.allocate_ghosted_array( field, FieldManager(1) );
        restart_file.read_view( std::string("fields/") + field, field_view.U);

        enum VarIndex_single { Ifield };
        UserData::FieldAccessor Ufield = U.getAccessor({{field, Ifield}});

        foreach_cell.foreach_cell( "restart_copy_field", U.getShape(),
          KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
        {
          Ufield.at( iCell, Ifield ) = field_view.at(iCell, 0);
        });
      }      
    }

    {     
      for( std::string particle_array : restart_file.list_fields("particles/") )
      {
        std::vector<hsize_t> field_dim = restart_file.get_field_layout(std::string("particles/")+particle_array+"/pos/x");

        DYABLO_ASSERT_HOST_RELEASE( field_dim.size() == 1, "Error while reading particles : hdf5 field particles/" << particle_array << "/pos/x is not a 1D array" );
        uint32_t nbParticles = field_dim[0];
        int mpi_rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();
        int nb_proc = GlobalMpiSession::get_comm_world().MPI_Comm_size();
        uint32_t nbParticles_local = (nbParticles*(mpi_rank+1))/nb_proc - (nbParticles*(mpi_rank))/nb_proc;

        U.new_ParticleArray(particle_array, nbParticles_local);
        const auto& P_pos = U.getParticleArray(particle_array);
        restart_file.read_view( std::string("particles/")+particle_array+"/pos/x", Kokkos::subview( P_pos.particle_position, Kokkos::ALL(), (int)IX ) );
        restart_file.read_view( std::string("particles/")+particle_array+"/pos/y", Kokkos::subview( P_pos.particle_position, Kokkos::ALL(), (int)IY ) );
        restart_file.read_view( std::string("particles/")+particle_array+"/pos/z", Kokkos::subview( P_pos.particle_position, Kokkos::ALL(), (int)IZ ) );

        ParticleData file_data( P_pos, FieldManager(1) );

        for( std::string attr : restart_file.list_fields(std::string("particles/")+particle_array+"/attributes"))
        {
          restart_file.read_view( std::string("particles/")+particle_array+"/attributes/"+attr, Kokkos::subview( file_data.particle_data, Kokkos::ALL(), 0 ) );

          U.new_ParticleAttribute( particle_array, attr );
          enum VarIndex_attr{ Iattr };
          UserData::ParticleAccessor Pwrite = U.getParticleAccessor( particle_array, {{attr, Iattr}} );
          foreach_particle.foreach_particle( "restart_copy_pos", file_data,
            KOKKOS_LAMBDA(const ForeachParticle::ParticleIndex& iPart)
          {
            Pwrite.at( iPart, Iattr ) = file_data.at_ivar(iPart, 0);
          });          
        }
      }      
    }
  }  
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, dyablo::InitialConditions_restart, "restart");