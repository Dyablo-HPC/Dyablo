#include "io/IOManager_base.h"

#include <cstdio>
#include <sstream>

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "amr/LightOctree.h"
#include "io/IOManager_base.h"
#include "utils/io/HDF5ViewWriter.h"
#include "foreach_cell/ForeachCell.h"

#include "utils/monitoring/Timers.h"

namespace dyablo { 

namespace{
  std::string main_xdmf_footer=R"xml(
    </Grid>
  </Domain>
</Xdmf>)xml";
}

class IOManager_hdf5 : public IOManager{
public: 
  IOManager_hdf5(
    ConfigMap& configMap,
    ForeachCell& foreach_cell,
    Timers& timers )
  : foreach_cell(foreach_cell),
    timers( timers ),
    filename( configMap.getValue<std::string>("output", "outputDir", "./") + "/" + configMap.getValue<std::string>("output", "outputPrefix", "output") )
  {
    std::string write_variables = configMap.getValue<std::string>("output", "write_variables", "rho" );
    std::stringstream sstream(write_variables);
    std::string var_name;
    while(std::getline(sstream, var_name, ','))
    { //use comma as delim for cutting string
      try{
        write_varindexes.insert( FieldManager::getiVar(var_name) );
      } catch (...) {
        std::cout << "WARNING : Output variable not found : '" << var_name << "'" << std::endl; 
      }      
    }
    
    { // Write main xdmf file with 0 timesteps
      // prepare suffix string
      main_xdmf_fd = fopen( (filename + "_main.xmf").c_str(), "w" );
      fprintf(main_xdmf_fd, 
R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
  <Domain Name="MainTimeSeries">
    <Grid Name="MainTimeSeries" GridType="Collection" CollectionType="Temporal">)xml");
      fprintf(main_xdmf_fd, "%s", main_xdmf_footer.c_str());
      fflush(main_xdmf_fd);
    }
  }
  ~IOManager_hdf5()
  {
    fclose(main_xdmf_fd);
  }

  void save_snapshot( const ForeachCell::CellArray_global_ghosted& U, uint32_t iter, real_t time );

  struct Data;
private:
  ForeachCell& foreach_cell;
  Timers& timers;
  std::string filename;
  std::set<VarIndex> write_varindexes;
  FILE* main_xdmf_fd;
};

void IOManager_hdf5::save_snapshot( const ForeachCell::CellArray_global_ghosted& U_, uint32_t iter, real_t time )
{
  static_assert( std::is_same_v<decltype(U_.U), DataArrayBlock>, "Only compatible with DataArrayBlock" );

  int ndim = foreach_cell.getDim();
  int nbNodesPerCell = (ndim-1)*4;

  uint32_t nbOcts_local = foreach_cell.get_amr_mesh().getNumOctants();
  uint64_t nbOcts_global = foreach_cell.get_amr_mesh().getGlobalNumOctants();
  uint32_t bx = U_.bx;
  uint32_t by = U_.by;
  uint32_t bz = U_.bz;
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
    base_filename = filename + strsuffix.str();
  }

  if( foreach_cell.get_amr_mesh().getRank() == 0 )
  { 


    // Append Current timestep to main xmdf file
    fseek( main_xdmf_fd, -main_xdmf_footer.size(), SEEK_END );
    fprintf(main_xdmf_fd,
R"xml(
      <xi:include href="%s" xpointer="xpointer(//Xdmf/Domain/Grid)" />)xml",
      (base_filename + ".xmf").c_str());
    fprintf(main_xdmf_fd, "%s", main_xdmf_footer.c_str());
    fflush(main_xdmf_fd);

    // Write xdmf file (only master MPI process)
    FILE* fd = fopen( (base_filename + ".xmf").c_str(), "w" );

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
      nbOcts_global*nbNodesPerOct, (int)sizeof(real_t),
      base_filename.c_str()
    );

    for( const VarIndex& iVar : write_varindexes )
    {
      if( U_.fm.enabled(iVar) )
      {
        fprintf(fd, 
R"xml(
      <Attribute Name="%s" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="%lu 1" NumberType="Float" Precision="%d" Format="HDF">
         %s.h5:/%s
        </DataItem>
      </Attribute>)xml",
          FieldManager::var_name(iVar).c_str(),
          global_num_cells, (int)sizeof(real_t),
          base_filename.c_str(), FieldManager::var_name(iVar).c_str()
        );
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
    auto linearize_iCell = KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell)
    {
      return iCell.i + iCell.bx * (iCell.j + iCell.by * ( iCell.k + iCell.bz * iCell.iOct.iOct )); 
    };

    HDF5ViewWriter hdf5_writer( base_filename + ".h5" );

    // Compute and write node coordinates
    {
      Kokkos::View< real_t**, Kokkos::LayoutLeft > coordinates("coordinates", 3, nbOcts_local*nbNodesPerOct);

      ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

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
        real_t oct_size = lmesh.getSize( {iOct, false} );
        coordinates(IX, index) = pos[IX] + (i * oct_size)/bx;
        coordinates(IY, index) = pos[IY] + (j * oct_size)/by;
        coordinates(IZ, index) = pos[IZ] + (k * oct_size)/bz;
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
    for( const VarIndex& iVar : write_varindexes )
    {
      if( U_.fm.enabled(iVar) )
      {
        std::string var_name = FieldManager::var_name(iVar).c_str();
        Kokkos::View< real_t*, Kokkos::LayoutLeft > tmp_view(var_name, local_num_cells);
        foreach_cell.foreach_cell( "compute_node_coordinates", U_,
        CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
        {
          uint32_t iCell_lin = linearize_iCell( iCell );
          tmp_view(iCell_lin) = U_.at(iCell, iVar);
        });
        hdf5_writer.collective_write( var_name, tmp_view );
      }
    }   
  }
}

}// namespace dyablo


FACTORY_REGISTER( dyablo::IOManagerFactory, dyablo::IOManager_hdf5, "IOManager_hdf5" );

