#pragma once 

#include <iostream>
#include <fstream>
#include <vector>

#include "amr/AMRmesh.h"

namespace dyablo{
namespace debug{

inline void output_vtk( const std::string& name, const AMRmesh& mesh )
{
  auto print_array = []( std::ostream& out, const auto& v )
  {
    for( const auto& e : v )
      out << e << " ";
    out << std::endl;
  };

  std::string filename = name+"_"+std::to_string(mesh.getRank())+".vtu";

  std::cout << "DEBUG OUPUT MESH " << filename << std::endl;
 
  uint32_t nofCubes = mesh.getNumOctants() + mesh.getNumGhosts();

  std::vector<int> cells_is_ghost(nofCubes);
  std::vector<int> cells_iOct(nofCubes);

  std::vector<double> nodes_Coordinates(nofCubes*8*3);
  std::vector<uint32_t> cells_Connectivity(nofCubes*8);
  std::vector<uint32_t> cells_offsets(nofCubes);
  std::vector<int> cells_types(nofCubes, 11);

  for(uint32_t i=0; i<nofCubes; i++)
  {
    cells_iOct[i] = i;

    real_t px, py, pz;
    real_t size_x, size_y, size_z;

    if( i<mesh.getNumOctants() )
    {
      px = mesh.getCoordinates(i)[0];
      py = mesh.getCoordinates(i)[1];
      pz = mesh.getCoordinates(i)[2];
      size_x = mesh.getSize(i)[0];
      size_y = mesh.getSize(i)[1];
      size_z = mesh.getSize(i)[2];
    }
    else
    {
      uint32_t iOct = i-mesh.getNumOctants();
      px = mesh.getCoordinatesGhost(iOct)[0];
      py = mesh.getCoordinatesGhost(iOct)[1];
      pz = mesh.getCoordinatesGhost(iOct)[2];
      size_x = mesh.getSizeGhost(iOct)[0];
      size_y = mesh.getSizeGhost(iOct)[1];
      size_z = mesh.getSizeGhost(iOct)[2];
      cells_is_ghost[i] = 1;
    }

    for( int16_t dz=0; dz<2; dz++ )
    for( int16_t dy=0; dy<2; dy++ )
    for( int16_t dx=0; dx<2; dx++ )
    {
      int di = dx + 2*dy + 4*dz;
      nodes_Coordinates[3*(8*i+di) + 0] = px + size_x * dx;
      nodes_Coordinates[3*(8*i+di) + 1] = py + size_y * dy;
      nodes_Coordinates[3*(8*i+di) + 2] = pz + size_z * dz;
      cells_Connectivity[8*i+di] = 8*i+di;
    }

    cells_offsets[i] = 8*i+8;
  }

  std::ofstream out( filename );
  out << "<?xml version=\"1.0\"?>"                                                      << std::endl;
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
  out << "  <UnstructuredGrid>"                                                         << std::endl;
  out << "    <Piece NumberOfCells=\"" << nofCubes << "\" NumberOfPoints=\"" << nofCubes*8 << "\">" << std::endl;
  out << "      <Points>"                                                               << std::endl;
  out << "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
  print_array(out, nodes_Coordinates);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </Points>"                                                              << std::endl;
  out << "      <CellData>"                                                             << std::endl;
  out << "        <DataArray type=\"Int32\" Name=\"is_ghost\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_is_ghost);
  out << "        </DataArray>"                                                         << std::endl;
  out << "        <DataArray type=\"Int32\" Name=\"iOct\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_iOct);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </CellData>"                                                            << std::endl;
  out << "      <Cells>"                                                                << std::endl;
  out << "        <DataArray type=\"UInt32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_Connectivity);
  out << "        </DataArray>"                                                         << std::endl;
  out << "        <DataArray type=\"UInt32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_offsets);
  out << "        </DataArray>"                                                         << std::endl;
  out << "        <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
  print_array(out, cells_types);
  out << "        </DataArray>"                                                         << std::endl;
  out << "      </Cells>"                                                               << std::endl;
  out << "    </Piece>"                                                                 << std::endl;
  out << "  </UnstructuredGrid>"                                                        << std::endl;
  out << "</VTKFile>"                                                                   << std::endl;

}

}
} //namespace dyablo