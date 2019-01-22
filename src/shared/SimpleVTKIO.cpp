#include "shared/SimpleVTKIO.h"

namespace euler_pablo {

// ==================================================
// ==================================================
void writeVTK(AMRmesh& amr_mesh,
	      std::string filenameSuffix,
	      DataArray data,
	      id2index_t  fm,
	      const ConfigMap& configMap)
{

  uint8_t dim = amr_mesh.getDim();
  
  // if done already, compute mesh connectivity
  // and store nodes coordinates
  //const auto& connectivity = ;
  if (amr_mesh.getConnectivity().size() == 0) {
    amr_mesh.computeConnectivity();
  }

  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
  
  std::stringstream name;
  name << outputPrefix
       << "-"  << std::setfill('0') << std::setw(4) << amr_mesh.getNproc()
       << "-p" << std::setfill('0') << std::setw(4) << amr_mesh.getRank()
       << "-"  << filenameSuffix << ".vtu";
  
  std::ofstream out(name.str().c_str());
  if(!out.is_open()){
    std::stringstream ss;
    ss << filenameSuffix << "*.vtu cannot be opened and it won't be written.";
    amr_mesh.getLog() << ss.str();
    return;
  }

  //auto nodes = amr_mesh.getNodes();
  int nofNodes = amr_mesh.getNodes().size();
  int nofOctants = amr_mesh.getConnectivity().size();
  int nofAll = nofOctants;
  out << "<?xml version=\"1.0\"?>" << std::endl
      << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl
      << "  <UnstructuredGrid>" << std::endl
      << "    <Piece NumberOfCells=\"" << amr_mesh.getConnectivity().size() << "\" NumberOfPoints=\"" << amr_mesh.getNodes().size() << "\">" << std::endl;
  out << "      <CellData Scalars=\"Data\">" << std::endl;
  out << "      <DataArray type=\"Float64\" Name=\"Data\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl
      << "          " << std::fixed;
  int ndata = amr_mesh.getConnectivity().size();
  for(int i = 0; i < ndata; i++)
    {
      out << std::setprecision(6) << i*123 << " ";  // data[i] << " ";
      if((i+1)%4==0 && i!=ndata-1)
	out << std::endl << "          ";
    }
  out << std::endl << "        </DataArray>" << std::endl
      << "      </CellData>" << std::endl
      << "      <Points>" << std::endl
      << "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\""<< 3 <<"\" format=\"ascii\">" << std::endl
      << "          " << std::fixed;
  for(int i = 0; i < nofNodes; i++)
    {
      for(int j = 0; j < 3; ++j){
	if (j==0) out << std::setprecision(6) << amr_mesh.getMap().mapX(amr_mesh.getNodes()[i][j]) << " ";
	if (j==1) out << std::setprecision(6) << amr_mesh.getMap().mapY(amr_mesh.getNodes()[i][j]) << " ";
	if (j==2) out << std::setprecision(6) << amr_mesh.getMap().mapZ(amr_mesh.getNodes()[i][j]) << " ";
      }
      if((i+1)%4==0 && i!=nofNodes-1)
	out << std::endl << "          ";
    }
  out << std::endl << "        </DataArray>" << std::endl
      << "      </Points>" << std::endl
      << "      <Cells>" << std::endl
      << "        <DataArray type=\"UInt64\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl
      << "          ";
  for(int i = 0; i < nofOctants; i++)
    {
      for(int j = 0; j < amr_mesh.getNnodes(); j++)
	{
	  int jj = j;
	  if (dim==2){
	    if (j<2){
	      jj = j;
	    }
	    else if(j==2){
	      jj = 3;
	    }
	    else if(j==3){
	      jj = 2;
	    }
	  }
	  out << amr_mesh.getConnectivity()[i][jj] << " ";
	}
      if((i+1)%3==0 && i!=nofOctants-1)
	out << std::endl << "          ";
    }
  out << std::endl << "        </DataArray>" << std::endl
      << "        <DataArray type=\"UInt64\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl
      << "          ";
  for(int i = 0; i < nofAll; i++)
    {
      out << (i+1)*amr_mesh.getNnodes() << " ";
      if((i+1)%12==0 && i!=nofAll-1)
	out << std::endl << "          ";
    }
  out << std::endl << "        </DataArray>" << std::endl
      << "        <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl
      << "          ";
  for(int i = 0; i < nofAll; i++)
    {
      int type;
      type = 5 + (dim*2);
      out << type << " ";
      if((i+1)%12==0 && i!=nofAll-1)
	out << std::endl << "          ";
    }
  out << std::endl << "        </DataArray>" << std::endl
      << "      </Cells>" << std::endl
      << "    </Piece>" << std::endl
      << "  </UnstructuredGrid>" << std::endl
      << "</VTKFile>" << std::endl;
  
  if(amr_mesh.getRank() == 0){
    name.str("");
    name << outputPrefix
	 << "-" << std::setfill('0') << std::setw(4) << amr_mesh.getNproc()
	 << "-" << filenameSuffix << ".pvtu";
    
    std::ofstream pout(name.str().c_str());
    if(!pout.is_open()){
      std::stringstream ss;
      ss << filenameSuffix << "*.pvtu cannot be opened and it won't be written." << std::endl;
      amr_mesh.getLog() << ss.str();
      return;
    }
    
    pout << "<?xml version=\"1.0\"?>" << std::endl
	 << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl
	 << "  <PUnstructuredGrid GhostLevel=\"0\">" << std::endl
	 << "    <PPointData>" << std::endl
	 << "    </PPointData>" << std::endl
	 << "    <PCellData Scalars=\"Data\">" << std::endl
	 << "      <PDataArray type=\"Float64\" Name=\"Data\" NumberOfComponents=\"1\"/>" << std::endl
	 << "    </PCellData>" << std::endl
	 << "    <PPoints>" << std::endl
	 << "      <PDataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\"/>" << std::endl
	 << "    </PPoints>" << std::endl;
    for(int i = 0; i < amr_mesh.getNproc(); i++)
      pout << "    <Piece Source=\""
	   << outputPrefix
	   << "-"  << std::setfill('0') << std::setw(4) << amr_mesh.getNproc()
	   << "-p" << std::setfill('0') << std::setw(4) << i
	   << "-"  << filenameSuffix << ".vtu\"/>" << std::endl;
    pout << "  </PUnstructuredGrid>" << std::endl
	 << "</VTKFile>";
    
    pout.close();
    
  }
#if BITPIT_ENABLE_MPI==1
  if (isCommSet()) {
    MPI_Barrier(m_comm);
  }
#endif
  
} // writeVTK

} // namespace euler_pablo

