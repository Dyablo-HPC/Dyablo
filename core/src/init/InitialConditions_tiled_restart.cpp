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
    for( int i=0; i<rank; i++ )
    {
      local_dims[i] = view.extent(rank-1-i);
      global_dims[i] = view.extent(rank-1-i);
      local_start[i] = 0;
    }

    hid_t filespace = H5Screate_simple( rank, global_dims, nullptr );
    hid_t  memspace = H5Screate_simple( rank,  local_dims, nullptr );


    // set some properties
    hid_t dataset_properties = H5Pcreate(H5P_DATASET_ACCESS);    

    hid_t dataset = H5Dopen2( m_hdf5_file, name.c_str(), dataset_properties );
    
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


class InitialConditions_tiled_restart : public InitialConditions{
    struct Data{
        ForeachCell& foreach_cell;
        
        int level_min, level_max;
    } data;
    std::string filename;
    ForeachParticle foreach_particle;

    // Limits of the virtual domain
    uint32_t nrep_x, nrep_y, nrep_z;
public:
  InitialConditions_tiled_restart(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : data({
      foreach_cell,
      foreach_cell.get_amr_mesh().get_level_min(),
      foreach_cell.get_amr_mesh().get_level_max()
    }),
    filename( configMap.getValue<std::string>( "restart", "filename", "restart.h5" ) ),
    foreach_particle(foreach_cell.get_amr_mesh(), configMap),
    nrep_x( configMap.getValue<int>("restart", "nrep_x", 1)),
    nrep_y( configMap.getValue<int>("restart", "nrep_y", 1)),
    nrep_z( configMap.getValue<int>("restart", "nrep_z", 1))
  {
    auto comm = foreach_cell.get_amr_mesh().getMpiComm();
    int nranks = comm.MPI_Comm_size();

    DYABLO_ASSERT_HOST_RELEASE(nranks == nrep_x*nrep_y*nrep_z, "When tiling a restart, the total number of tiles has to be the exact number of MPI ranks");
  }

  void init( UserData& U )
  {
    ForeachCell& foreach_cell = data.foreach_cell;
    AMRmesh& pmesh     = foreach_cell.get_amr_mesh();

    uint32_t ndim = pmesh.getDim();
    int level_min = data.level_min;
    int level_max = data.level_max;

    uint32_t nrep_x = this->nrep_x;
    uint32_t nrep_y = this->nrep_y;
    uint32_t nrep_z = this->nrep_z;

    restart_file restart_file(filename);
    uint32_t nOcts_written;

    // Initialize AMR tree
    {
        LightOctree_storage<>::oct_data_t lmesh_data;
        restart_file.read_view_init<decltype(lmesh_data),0>( "Octree", lmesh_data );

        uint32_t nbCells_local = lmesh_data.extent(0);
        uint32_t nbCells_ghost = 0;

        constexpr real_t one_plus_eps = 1.00001;
        int ldiff_x = static_cast<int>(std::log2(nrep_x * one_plus_eps));
        int ldiff_y = static_cast<int>(std::log2(nrep_y * one_plus_eps));
        int ldiff_z = static_cast<int>(std::log2(nrep_z * one_plus_eps));

        int level_diff = std::max({ldiff_x, ldiff_y, ldiff_z});
        int old_lmin = level_min - level_diff;

        std::cout << "Restart: old level min detected is : " << old_lmin << std::endl;
        
        // Calculating coarse_oct_resolution of the original file
        // Oof ... Copying from LightOctree_storage.h. Maybe expose this as a public member in the class ?
        enum oct_data_field_t {
          ICORNERX, 
          ICORNERY, 
          ICORNERZ,
          ILEVEL
        };

        real_t lvl_min_size = 1.0 / (1U << old_lmin);
        uint32_t old_cor_x = 0;
        uint32_t old_cor_y = 0;
        uint32_t old_cor_z = 0;

        Kokkos::parallel_reduce("Restart, Coarse_oct_reduce", nbCells_local, 
          KOKKOS_LAMBDA(uint32_t iOct, uint32_t &cor_loc_x, uint32_t &cor_loc_y, uint32_t &cor_loc_z) {
            real_t oct_size = 1.0 / (1U << static_cast<uint32_t>(lmesh_data(iOct, ILEVEL)));
            uint32_t Nx = one_plus_eps * (lmesh_data(iOct, ICORNERX) * oct_size + lvl_min_size) / lvl_min_size;
            uint32_t Ny = one_plus_eps * (lmesh_data(iOct, ICORNERY) * oct_size + lvl_min_size) / lvl_min_size;
            uint32_t Nz = one_plus_eps * (lmesh_data(iOct, ICORNERZ) * oct_size + lvl_min_size) / lvl_min_size;

            cor_loc_x = (Nx > cor_loc_x ? Nx : cor_loc_x);
            cor_loc_y = (Ny > cor_loc_y ? Ny : cor_loc_y);
            cor_loc_z = (Nz > cor_loc_z ? Nz : cor_loc_z);
          }, 
          Kokkos::Max<uint32_t>(old_cor_x),
          Kokkos::Max<uint32_t>(old_cor_y),
          Kokkos::Max<uint32_t>(old_cor_z));

        std::cout << "Restart: old coarse size detected : " << old_cor_x << " " << old_cor_y << " " << old_cor_z << std::endl;

        // Todo : Find this from input data ?
        const Kokkos::Array<uint32_t,3>& coarse_grid_size {old_cor_x, old_cor_y, old_cor_z};
        LightOctree_storage<> lmesh_storage( ndim, nbCells_local, nbCells_ghost, old_lmin, coarse_grid_size ); 


        Kokkos::deep_copy( lmesh_storage.getLocalSubview(), lmesh_data );

        // Read mesh from restart file
        Kokkos::Array<bool,3> periodic{false, false, false};
        LightOctree_hashmap input_lmesh( std::move(lmesh_storage), old_lmin, level_max, periodic );

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

                // Applying mod to roll over repetitions along each axis 
                pos[IX] = FMOD(pos[IX]*nrep_x, 1.0);
                pos[IY] = FMOD(pos[IY]*nrep_y, 1.0);
                if (ndim==3)
                  pos[IZ] = FMOD(pos[IZ]*nrep_z, 1.0);

                LightOctree::OctantIndex iOct_input = input_lmesh.getiOctFromPos( pos );
                int level_input = input_lmesh.getLevel(iOct_input);

                markers(iOct_new) = ( level_input + level_diff > level );
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

      nOcts_written = input_lmesh.getNumOctants();
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
        
        const auto shape = U.getShape();
        const uint32_t bx = shape.bx;
        const uint32_t by = shape.by;
        const uint32_t bz = shape.bz;
        const uint32_t nCells_written = nOcts_written * bx * by * bz;
        
        Kokkos::View<real_t*> field_view("hdf5_view", nCells_written);
        restart_file.read_view( std::string("fields/") + field, field_view);

        enum VarIndex_single { Ifield };
        UserData::FieldAccessor Ufield = U.getAccessor({{field, Ifield}});

        foreach_cell.foreach_cell( "restart_copy_field", U.getShape(),
          KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
        {
          auto iCell_mod = iCell;
          iCell_mod.iOct.iOct = iCell_mod.iOct.iOct % nOcts_written;
          uint32_t index = iCell_mod.iOct.iOct * bz*by*bx + iCell_mod.k*bx*by + iCell_mod.j*bx + iCell_mod.i;
          Ufield.at( iCell, Ifield ) = field_view(index);
        });
      }      
    }

    { bool warning_printed = false;
      for( std::string particle_array : restart_file.list_fields("particles/") )
      {
        if (!warning_printed) {
          std::cout << "WARNING : Tiled restart does not work with particles yet !" << std::endl;
          warning_printed = true;
        }

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

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_tiled_restart, 
                 "tiled_restart");