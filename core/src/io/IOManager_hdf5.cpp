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
      write_varindexes.insert(FieldManager::getiVar(var_name));
    }
    
    { // Write main xdmf file with 0 timesteps
      // prepare suffix string
      main_xdmf_fd = fopen( (filename + "_main.xmf").c_str(), "w" );
      fprintf(main_xdmf_fd, 
R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
  <Domain Name="MainTimeSeries">
    <Grid Name="MainTimeSeries" GridType="Collection" CollectionType="Temporal">
    </Grid>
  </Domain>
</Xdmf>)xml");
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
  const ForeachCell& foreach_cell;
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
    fseek( main_xdmf_fd, -32, SEEK_END );
    fprintf(main_xdmf_fd,
R"xml(
      <xi:include href="%s" xpointer="xpointer(//Xdmf/Domain/Grid)" />)xml",
      (base_filename + ".xmf").c_str());
    fprintf(main_xdmf_fd, 
R"xml(
    </Grid>
  </Domain>
</Xdmf>)xml");
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
      global_num_cells*nbNodesPerCell, (int)sizeof(real_t),
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

    uint32_t local_num_cells = foreach_cell.getNumCells();

    HDF5ViewWriter hdf5_writer( base_filename + ".h5" );

    // Compute and write node coordinates
    {
      Kokkos::View< real_t**, Kokkos::LayoutLeft > coordinates("coordinates", 3, nbNodesPerCell*local_num_cells);

      ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

      foreach_cell.foreach_cell( "compute_node_coordinates", U_,
        CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
      {
        auto pos = cells.getCellCenter(iCell);
        auto size = cells.getCellSize(iCell);

        uint32_t iCell_lin = linearize_iCell( iCell );
        for(int k=0; k<(ndim-1); k++)
        for(int j=0; j<2; j++)
        for(int i=0; i<2; i++)
        {
          int iNode = i+2*j+4*k;
          coordinates( IX, iCell_lin*nbNodesPerCell + iNode ) = pos[IX]+(i-0.5)*size[IX];
          coordinates( IY, iCell_lin*nbNodesPerCell + iNode ) = pos[IY]+(j-0.5)*size[IY];
          coordinates( IZ, iCell_lin*nbNodesPerCell + iNode ) = pos[IZ]+(k-0.5)*size[IZ];
        }

      });

      hdf5_writer.collective_write( "coordinates", coordinates );
    }

    // Compute and write node connectivity
    {
      uint64_t first_cell = foreach_cell.get_amr_mesh().getGlobalIdx(0)*U_.bx*U_.by*U_.bz;

      Kokkos::View< uint64_t**, Kokkos::LayoutLeft > connectivity("connectivity", nbNodesPerCell, local_num_cells);

      Kokkos::parallel_for( "fill_connectivity", local_num_cells*nbNodesPerCell,
        KOKKOS_LAMBDA( uint64_t i )
      {
        constexpr int node_shuffle[] = {0,1,3,2,4,5,7,6};

        uint64_t iNode = i%nbNodesPerCell;
        uint64_t iCell = i/nbNodesPerCell;

        connectivity(iNode, iCell) =  (first_cell+iCell)*nbNodesPerCell + node_shuffle[iNode];
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

