#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "userdata_utils.h"

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
  restart_file(const std::string& filename)
  {
    hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
#ifdef DYABLO_USE_MPI
    H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
#endif
    m_hdf5_file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist);
    H5Pclose(plist);
  }

  template< typename T, int iOct_pos=T::rank-1  >
  void read_view_init( const std::string& name, T& view )
  {
    hid_t hdf5_type = get_hdf5_type<typename T::value_type>();
    constexpr hid_t rank = T::rank;

    hid_t dataset_properties = H5Pcreate(H5P_DATASET_ACCESS);
    hid_t dataset = H5Dopen2( m_hdf5_file, name.c_str(), dataset_properties );
    hid_t filespace = H5Dget_space( dataset );

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
    constexpr hid_t rank = T::rank;
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
    filename( configMap.getValue<std::string>( "restart", "filename", "restart.h5" ) )
  {}

  void init( ForeachCell::CellArray_global_ghosted& U, const FieldManager& fieldMgr )
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
        LightOctree_storage<> lmesh_storage( ndim, nbCells_local, nbCells_ghost );

        Kokkos::deep_copy( lmesh_storage.getLocalSubview(), lmesh_data );

        // Read mesh from restart file
        Kokkos::Array<bool,3> periodic{false, false, false};
        LightOctree_hashmap input_lmesh( std::move(lmesh_storage), level_min, level_max, periodic );

        std::cout << "Restart mesh : " << input_lmesh.getNumOctants() << " octs." << std::endl;
        
        // Refine to level_min
        for (uint8_t level=0; level<level_min; ++level)
        {
            pmesh.adaptGlobalRefine(); 
        } 
        pmesh.loadBalance();

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
                real_t oct_size = lmesh.getSize({iOct_new, false});
                pos[IX] += oct_size*0.01; // Add epsilon to avoid hitting boundary
                pos[IY] += oct_size*0.01;
                if(ndim==3)
                  pos[IZ] += oct_size*0.01;

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
        // TODO wrap reallocation in U.reallocate(pmesh)?
        U  = foreach_cell.allocate_ghosted_array( "U" , fieldMgr );

        restart_file.read_view( "U", U.U );        
    }
  }  
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, dyablo::InitialConditions_restart, "restart");