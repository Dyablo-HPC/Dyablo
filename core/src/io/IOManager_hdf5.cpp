#include "io/IOManager_base.h"

#include <cstdio>
#include <sstream>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "io/IOManager_base.h"
#include "utils/io/HDF5ViewWriter.h"
#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

#include "utils/monitoring/Timers.h"
#include "utils/config/named_enum.h"

#include "filesystem"

enum OutputRealType {
  OT_FLOAT, 
  OT_DOUBLE
};

template<>
inline named_enum<OutputRealType>::init_list named_enum<OutputRealType>::names()
{
  return{
    {OutputRealType::OT_FLOAT, "float"},
    {OutputRealType::OT_DOUBLE, "double"}
  };
}

namespace dyablo { 

namespace{

class MainXmfFile{
public:
  std::string filename;
  FILE* main_xdmf_fd = nullptr;
  bool file_created = false;

  MainXmfFile(const std::string& filename)
  : filename(filename)
  {}

  ~MainXmfFile()
  {
    if(main_xdmf_fd)
      fclose(main_xdmf_fd);
  }

  MainXmfFile( ) = default;
  MainXmfFile( MainXmfFile&& ) = default;
  MainXmfFile& operator=( MainXmfFile&& ) = default;

  static constexpr std::string_view main_xdmf_footer=R"xml(
    </Grid>
  </Domain>
</Xdmf>)xml";

  void create_file()
  { 
    namespace fs = std::filesystem;
    fs::create_directories( fs::path(filename).remove_filename() );
    
    // Write main xdmf file with 0 timesteps
    // prepare suffix string
    this->main_xdmf_fd = fopen( filename.c_str(), "w" );
    fprintf(main_xdmf_fd, 
R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
  <Domain Name="MainTimeSeries">
    <Grid Name="MainTimeSeries" GridType="Collection" CollectionType="Temporal">)xml");
    fprintf(main_xdmf_fd, "%s", std::string(main_xdmf_footer).c_str());
    fflush(main_xdmf_fd);
  }

  void append_xmf(const std::string& xmf_filename)
  {
    if(!file_created)
    {
      create_file();
      file_created = true;
    }

    // Append Current timestep to main xmdf file
    fseek( main_xdmf_fd, -main_xdmf_footer.size(), SEEK_END );
    fprintf(main_xdmf_fd,
R"xml(
      <xi:include href="%s" xpointer="xpointer(//Xdmf/Domain/Grid)" />)xml",
      xmf_filename.c_str());
    fprintf(main_xdmf_fd, "%s", std::string(main_xdmf_footer).c_str());
    fflush(main_xdmf_fd);
  }
};

}

class IOManager_hdf5 : public IOManager{
public: 
  IOManager_hdf5(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers )
  : foreach_cell(foreach_cell),
    timers( timers ),
    output_dir(configMap.getValue<std::string>("output", "outputDir", ".")),
    filename_prefix( configMap.getValue<std::string>("output", "outputPrefix", "output") ),
    xmin(configMap.getValue<real_t>("mesh", "xmin", 0.0)),
    xmax(configMap.getValue<real_t>("mesh", "xmax", 1.0)),
    ymin(configMap.getValue<real_t>("mesh", "ymin", 0.0)),
    ymax(configMap.getValue<real_t>("mesh", "ymax", 1.0)),
    zmin(configMap.getValue<real_t>("mesh", "zmin", 0.0)),
    zmax(configMap.getValue<real_t>("mesh", "zmax", 1.0)),
    output_real_t(configMap.getValue<OutputRealType>("output", "output_real_type", OT_FLOAT))
  {
    {
      std::string write_variables = configMap.getValue<std::string>("output", "write_variables", "rho" );
      std::stringstream sstream(write_variables);
      std::string var_name;
      while(std::getline(sstream, var_name, ','))
      { //use comma as delim for cutting string
        write_varnames.insert(var_name);
      }
    }
    {
      std::string write_variables = configMap.getValue<std::string>("output", "write_particle_variables", "" );
      std::stringstream sstream(write_variables);
      std::string var_name;
      while(std::getline(sstream, var_name, ','))
      { //use comma as delim for cutting string

        size_t slashPos = var_name.find_last_of("/");
        if( slashPos == std::string::npos )
        {
          write_particle_attributes[var_name]; // Create empty set for var_name
          continue;
        }
        
        auto trim = [](std::string& str)
        {
          str.erase(std::remove(str.begin(),str.end(),' '),str.end());
        };

        std::string array_name = var_name.substr(0, slashPos);
        trim(array_name);
        std::string attr_name = var_name.substr(slashPos + 1);
        trim(attr_name);

        //std::cout << "parsed array attribute '" << array_name << "' / '" << attr_name << "'" << std::endl;

        write_particle_attributes[array_name].insert( attr_name );      
      }
    }

    std::string filepath = output_dir + "/" + filename_prefix;
    main_xdmf_fd = MainXmfFile( filepath + "_main.xmf" );
  }

  void save_snapshot( const UserData& U_, ScalarSimulationData& scalar_data )
  {
    switch (output_real_t) {
    case OutputRealType::OT_FLOAT:  save_snapshot_aux<float>(U_, scalar_data); break;
    case OutputRealType::OT_DOUBLE: save_snapshot_aux<double>(U_, scalar_data); break;
  }
  }
  template <typename output_real_t>
  void save_snapshot_aux( const UserData& U_, ScalarSimulationData& scalar_data );
  template <typename output_real_t>
  void save_particles( const UserData& U, const std::string& particle_array,  uint32_t iter, real_t time );

  struct Data;
private:
  ForeachCell& foreach_cell;
  Timers& timers;
  std::string output_dir;
  std::string filename_prefix;
  std::set<std::string> write_varnames;
  std::map<std::string, std::set<std::string>> write_particle_attributes;
  MainXmfFile main_xdmf_fd;
  std::map<std::string, MainXmfFile> particles_main_xdmf_fds;
  const real_t xmin, xmax;
  const real_t ymin, ymax;
  const real_t zmin, zmax;

  OutputRealType output_real_t;
};

namespace{

template< typename T > 
std::string xmf_type_attr()
{
  static_assert(!std::is_same_v<T,T>, "Please define a specialization for xmf_type_attr for this type");
  return "";
}

template<> [[maybe_unused]] std::string xmf_type_attr<int32_t>   () { return R"xml(NumberType="Int"   Precision="4")xml"; }
template<> [[maybe_unused]] std::string xmf_type_attr<int64_t>   () { return R"xml(NumberType="Int"   Precision="8")xml"; }
template<> [[maybe_unused]] std::string xmf_type_attr<uint32_t>  () { return R"xml(NumberType="UInt"  Precision="4")xml"; }
template<> [[maybe_unused]] std::string xmf_type_attr<uint64_t>  () { return R"xml(NumberType="UInt"  Precision="8")xml"; }
template<> [[maybe_unused]] std::string xmf_type_attr<float>     () { return R"xml(NumberType="Float" Precision="4")xml"; }
template<> [[maybe_unused]] std::string xmf_type_attr<double>    () { return R"xml(NumberType="Float" Precision="8")xml"; }

} // namespace

template <typename output_real_t>
void IOManager_hdf5::save_snapshot_aux( const UserData& U_, ScalarSimulationData& scalar_data )
{
  int iter = scalar_data.get<int>( "iter" );
  real_t time = scalar_data.get<real_t>( "time" );

  //static_assert( std::is_same_v<decltype(U_.U), DataArrayBlock>, "Only compatible with DataArrayBlock" );

  int ndim = foreach_cell.getDim();
  int nbNodesPerCell = (ndim-1)*4;

  uint32_t nbOcts_local = foreach_cell.get_amr_mesh().getNumOctants();
  uint64_t nbOcts_global = foreach_cell.get_amr_mesh().getGlobalNumOctants();
  
  static_assert( ForeachCell::has_blocks() );
  uint32_t bx = foreach_cell.blockSize()[IX];
  uint32_t by = foreach_cell.blockSize()[IY];
  uint32_t bz = foreach_cell.blockSize()[IZ];
  uint32_t nbCellsPerOct = bx*by*bz;
  uint32_t Bx = bx+1;
  uint32_t By = by+1;
  uint32_t Bz = bz+1;
  uint32_t nbNodesPerOct = Bx*By*Bz;
  
  const LightOctree& lmesh = foreach_cell.get_amr_mesh().getLightOctree();

  std::string base_filename;
  {
    // prepare suffix string
    std::ostringstream strsuffix;
    strsuffix << "_iter";
    strsuffix.width(7);
    strsuffix.fill('0');
    strsuffix << iter;
    strsuffix.str();
    base_filename = filename_prefix + strsuffix.str();
  }

  if( foreach_cell.get_amr_mesh().getRank() == 0 )
  { 
    // Append Current timestep to main xmdf file
    main_xdmf_fd.append_xmf( base_filename + ".xmf");

    // Write xdmf file (only master MPI process)
    FILE* fd = fopen( (output_dir + "/" + base_filename + ".xmf").c_str(), "w" );

    uint64_t global_num_cells = foreach_cell.getNumCells_global();

    fprintf(fd, 
R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="%s" GridType="Uniform">
      <Time TimeType="Single" Value="%g" />
      <Topology TopologyType="%s" NumberOfElements="%lu">
        <DataItem Dimensions="%lu %d" DataType="UInt" Precision="8" Format="HDF">
          %s.h5:/connectivity
        </DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="%lu 3" NumberType="Float" Precision="%d" Format="HDF">
          %s.h5:/coordinates
        </DataItem>
      </Geometry>)xml",
      base_filename.c_str(), 
      time,
      ndim==2?"Quadrilateral":"Hexahedron", global_num_cells,
      global_num_cells, nbNodesPerCell,
      base_filename.c_str(),
      nbOcts_global*nbNodesPerOct, (int)sizeof(output_real_t),
      base_filename.c_str()
    );

    for( const std::string& var_name : write_varnames )
    {
      auto output_attr_xml = [&]( const std::string& type_str )
      {
        fprintf(fd, 
R"xml(
      <Attribute Name="%s" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="%lu 1" %s Format="HDF">
         %s.h5:/%s
        </DataItem>
      </Attribute>)xml",
          var_name.c_str(),
          global_num_cells, type_str.c_str(),
          base_filename.c_str(), var_name.c_str()
        );
      };

      if( var_name == "ioct" )
      {
        output_attr_xml(xmf_type_attr<uint64_t>());       
      }
      else if( var_name == "level" )
      {
        output_attr_xml(xmf_type_attr<LightOctree::level_t>());       
      }
      else if( var_name == "rank" )
      {
        output_attr_xml(xmf_type_attr<int>());       
      }
      else if( U_.has_field(var_name) )
      {
        output_attr_xml(xmf_type_attr<output_real_t>());
      }
      else
      {
        std::cout << "WARNING : Output variable requested but not enabled : '" << var_name << "'" << std::endl; 
      } 
    }

    fprintf(fd, 
R"xml(
    </Grid>
  </Domain>
</Xdmf>
)xml");

    fclose(fd);
  }

  { 
    // Write hdf5 file 
    auto linearize_iCell = KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
      return iCell.i + iCell.bx * (iCell.j + iCell.by * ( iCell.k + iCell.bz * iCell.iOct.iOct )); 
    };

    HDF5ViewWriter hdf5_writer( output_dir + "/" + base_filename + ".h5" );

    scalar_data.foreach_var( [&]( const std::string& name, auto val )
    {
      hdf5_writer.write_scalar(std::string("scalar_data/")+name, val);
    });

    // Compute and write node coordinates
    {
      Kokkos::View< output_real_t**, Kokkos::LayoutLeft > coordinates("coordinates", 3, nbOcts_local*nbNodesPerOct);

      ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

      const real_t xmin = this->xmin;
      const real_t ymin = this->ymin;
      const real_t zmin = this->zmin;
      const real_t Lx = xmax-xmin;
      const real_t Ly = ymax-ymin;
      const real_t Lz = zmax-zmin;

      // Create (bx+1)*(by+1)*(bz+1) nodes per octant
      Kokkos::parallel_for( "compute_coordinates", nbOcts_local*nbNodesPerOct,
        KOKKOS_LAMBDA( uint64_t index )
      {
        uint32_t iOct = index/nbNodesPerOct;
        uint32_t iNode = index%nbNodesPerOct;
        int k = iNode/(Bx*By);
        int j = (iNode/Bx)%By;
        int i = iNode%Bx;

        auto pos = lmesh.getCorner( {iOct, false} );
        auto oct_size = lmesh.getSize( {iOct, false} );
        coordinates(IX, index) = static_cast<output_real_t>(xmin + (pos[IX] + (i * oct_size[IX])/bx) * Lx);
        coordinates(IY, index) = static_cast<output_real_t>(ymin + (pos[IY] + (j * oct_size[IY])/by) * Ly);
        coordinates(IZ, index) = static_cast<output_real_t>(zmin + (pos[IZ] + (k * oct_size[IZ])/bz) * Lz);
      });

      hdf5_writer.collective_write( "coordinates", coordinates );
    }

    uint32_t local_num_cells = foreach_cell.getNumCells();

    // Compute and write node connectivity
    {
      uint64_t first_iOct = foreach_cell.get_amr_mesh().getGlobalIdx(0);

      Kokkos::View< uint64_t**, Kokkos::LayoutLeft > connectivity("connectivity", nbNodesPerCell, local_num_cells);

      Kokkos::parallel_for( "fill_connectivity", local_num_cells*nbNodesPerCell,
        KOKKOS_LAMBDA( uint64_t i )
      {
        uint32_t iNode_cell = i%nbNodesPerCell;        
        uint64_t iCell = i/nbNodesPerCell;
        uint32_t iOct = iCell/nbCellsPerOct;
        uint32_t iNode_block = iCell%nbCellsPerOct;

        // Compute logical position of cell corner in [0,1]^3 
        // node_shuffle to list nodes in right order for paraview
        constexpr uint32_t node_shuffle[8] = {0,1,3,2,4,5,7,6};
        uint32_t iNode_cell_shuffle = node_shuffle[iNode_cell];
        uint32_t i_cell = iNode_cell_shuffle%2;
        uint32_t j_cell = (iNode_cell_shuffle/2)%2;
        uint32_t k_cell = iNode_cell_shuffle/4;
        // Compute logical position of cell in block
        uint32_t i_block = iNode_block%bx;
        uint32_t j_block = (iNode_block/bx)%by;
        uint32_t k_block = iNode_block/(bx*by);          

        connectivity(iNode_cell, iCell) = (first_iOct+iOct)*nbNodesPerOct 
                                        + (i_cell+i_block) + (j_cell+j_block)*Bx + (k_cell+k_block)*Bx*By ;
      });

      hdf5_writer.collective_write( "connectivity", connectivity );
    }

    // Write selected variables
    for( const std::string& var_name : write_varnames )
    {
      if( var_name == "ioct" )
      {
        uint64_t first_local_iOct = foreach_cell.get_amr_mesh().getGlobalIdx(0);
        Kokkos::View< uint64_t*, Kokkos::LayoutLeft> ioct("ioct", local_num_cells);
        Kokkos::parallel_for( "fill_ioct", local_num_cells,
          KOKKOS_LAMBDA( uint32_t i )
        {
          uint64_t iOct = first_local_iOct + i/nbCellsPerOct;
          ioct(i) = iOct; 
        });
        hdf5_writer.collective_write("ioct", ioct);
      }
      else if( var_name == "level" )
      {
        Kokkos::View< LightOctree::level_t*, Kokkos::LayoutLeft> level("level", local_num_cells);
        Kokkos::parallel_for( "fill_level", local_num_cells,
          KOKKOS_LAMBDA( uint32_t i )
        {
          uint32_t iOct = i/nbCellsPerOct;
          level(i) = lmesh.getLevel( {iOct, false} );
        });
        hdf5_writer.collective_write("level", level);
      }
      else if( var_name == "rank" )
      {
        int local_rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();
        Kokkos::View< int*, Kokkos::LayoutLeft> rank("rank", local_num_cells);
        Kokkos::parallel_for( "fill_rank", local_num_cells,
          KOKKOS_LAMBDA( uint32_t i )
        {
          rank(i) = local_rank;
        });
        hdf5_writer.collective_write("rank", rank);
      }
      else
      { 
        if( U_.has_field(var_name) )
        { 
          Kokkos::View< output_real_t*, Kokkos::LayoutLeft > tmp_view(var_name, local_num_cells);
          
          auto field_view = U_.getField( var_name );
          foreach_cell.foreach_cell( "compute_node_coordinates", field_view,
          KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
          {
            uint32_t iCell_lin = linearize_iCell( iCell );
            tmp_view(iCell_lin) = static_cast<output_real_t>(field_view.at_ivar(iCell, 0));
          });
          hdf5_writer.collective_write( var_name, tmp_view );
        }
      }
    }   
  }

  for( const auto& p : write_particle_attributes )
  {
    std::string particle_array = p.first;

    if( U_.has_ParticleArray(particle_array) )
    {
      if( particles_main_xdmf_fds.find(particle_array) == particles_main_xdmf_fds.end() )
        particles_main_xdmf_fds[particle_array] = MainXmfFile(output_dir + "/" + filename_prefix + "_particles_" + particle_array + "_main.xmf");
      
      save_particles<output_real_t>( U_, particle_array, iter, time);
    }
    else
      std::cout << "WARNING : Output particle array requested but not enabled : '" << particle_array << "'" << std::endl;
  }
}

template <typename output_real_t>
void IOManager_hdf5::save_particles( const UserData& U, const std::string& particle_array, uint32_t iter, real_t time )
{
  std::string base_filename;
  {
    // prepare suffix string
    std::ostringstream strsuffix;
    strsuffix << "_particles_" << particle_array << "_iter";
    strsuffix.width(7);
    strsuffix.fill('0');
    strsuffix << iter;
    strsuffix.str();
    base_filename = filename_prefix + strsuffix.str();
  }

  auto mpi_comm = foreach_cell.get_amr_mesh().getMpiComm();

  uint32_t local_num_particles = U.getParticleArray(particle_array).getNumParticles();
  uint64_t global_num_particles;
  {
    uint64_t nbPart = local_num_particles;
    mpi_comm.MPI_Allreduce( &nbPart, &global_num_particles, 1, MpiComm::MPI_Op_t::SUM );
  }

  if( mpi_comm.MPI_Comm_rank() == 0 )
  { 
    // Append Current timestep to main xmdf file
    particles_main_xdmf_fds.at(particle_array).append_xmf(base_filename + ".xmf");

    // Write xdmf file (only master MPI process)
    FILE* fd = fopen( (output_dir + "/" + base_filename + ".xmf").c_str(), "w" );

    fprintf(fd, 
R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
  <Domain>
    <Grid Name="%s" GridType="Uniform">
      <Time TimeType="Single" Value="%g" />
      <Topology TopologyType="Polyvertex" NumberOfElements="%lu" />
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="%lu 3" NumberType="Float" Precision="%d" Format="HDF">
          %s.h5:/coordinates
        </DataItem>
      </Geometry>)xml",
      base_filename.c_str(), 
      time,
      global_num_particles,
      global_num_particles, (int)sizeof(output_real_t),
      base_filename.c_str()
    );

    for( const std::string& var_name : write_particle_attributes.at(particle_array) )
    {
      if( U.has_ParticleAttribute(particle_array, var_name) )
      {
        fprintf(fd, 
R"xml(
      <Attribute Name="%s" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="%lu 1" NumberType="Float" Precision="%d" Format="HDF">
         %s.h5:/%s
        </DataItem>
      </Attribute>)xml",
          var_name.c_str(),
          global_num_particles, (int)sizeof(output_real_t),
          base_filename.c_str(), var_name.c_str()
        );
      }
      else
      {
        std::cout << "WARNING : Output attribute requested but not enabled : '" << particle_array << "/" << var_name << "'" << std::endl; 
      }
    }

    fprintf(fd, 
R"xml(
    </Grid>
  </Domain>
</Xdmf>
)xml");

    fclose(fd);
  }

 { // Write hdf5 file
    HDF5ViewWriter hdf5_writer( output_dir + "/" + base_filename + ".h5" );

    { // Write coordinates

      const ParticleArray& P = U.getParticleArray(particle_array);
      Kokkos::View< output_real_t**, Kokkos::LayoutLeft > tmp_view("particles_coordinates", 3, local_num_particles);
      Kokkos::parallel_for( "compute_particles_coordinates", local_num_particles,
      KOKKOS_LAMBDA( uint32_t iPart )
      {
        tmp_view(IX, iPart) = static_cast<output_real_t>(P.pos(iPart,IX));
        tmp_view(IY, iPart) = static_cast<output_real_t>(P.pos(iPart,IY));
        tmp_view(IZ, iPart) = static_cast<output_real_t>(P.pos(iPart,IZ));
      });
      hdf5_writer.collective_write( "coordinates", tmp_view );
    } 

    // Write selected variables
    for( const std::string& var_name : write_particle_attributes.at(particle_array) )
    {
      if(  U.has_ParticleAttribute(particle_array, var_name) )
      {
        enum VarIndex {Ivar};
        auto Pin = U.getParticleAccessor(particle_array, {{var_name, Ivar}} );

        Kokkos::View< output_real_t*, Kokkos::LayoutLeft > tmp_view(particle_array + "/" + var_name, local_num_particles);
        Kokkos::parallel_for( "copy_particle_attribute", local_num_particles,
        KOKKOS_LAMBDA( uint32_t iPart )
        {
          tmp_view(iPart) = static_cast<output_real_t>(Pin.at(iPart,Ivar));
        });

        hdf5_writer.collective_write( var_name, tmp_view );
      }
    }   
 }
}

}// namespace dyablo


FACTORY_REGISTER( dyablo::IOManagerFactory, dyablo::IOManager_hdf5, "IOManager_hdf5" );

