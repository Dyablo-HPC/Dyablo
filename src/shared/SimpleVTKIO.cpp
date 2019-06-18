#include "shared/SimpleVTKIO.h"

namespace euler_pablo {

// =======================================================
// =======================================================
int build_var_to_write_map(str2int_t        & map,
			   const HydroParams& params,
			   const ConfigMap  & configMap)
{

  // Read parameter file and get the list of variable field names
  // we wish to write 
  // variable names must be comma-separated (no quotes !)
  // e.g. write_variables=rho,vx,vy,unknown
  std::string write_variables = configMap.getString("output", "write_variables", "rho,rho_vx");

  // second retrieve available / allowed names
  FieldManager fieldMgr;
  fieldMgr.setup(params, configMap);
  
  str2int_t avail_names = fieldMgr.get_names2index();
  
  // now tokenize
  std::istringstream iss(write_variables);
  std::string token;
  while (std::getline(iss, token, ',')) {

    // check if token is valid, i.e. present in avail_names
    auto got = avail_names.find(token);
    //std::cout << " " << token << " " << got->second << "\n";
    
    // if token is valid, we insert it into map
    if (got != avail_names.end()) {
      map[token] = got->second;
    }
    
  }
  
  return map.size();
    
} // build_var_to_write_map

// ==================================================
// ==================================================
void writeVTK(AMRmesh&         amr_mesh,
	      std::string      filenameSuffix,
	      DataArray        data,
	      id2index_t       fm,
	      str2int_t        names2index,
	      const ConfigMap& configMap,
              std::string      nameSuffix)
{

  // copy data from device to host
  DataArrayHost datah = Kokkos::create_mirror(data);
  // copy device data to host
  Kokkos::deep_copy(datah, data);

  // dimension : 2 or 3 ?
  uint8_t dim = amr_mesh.getDim();
  
  // if not done already, compute mesh connectivity
  // and store nodes coordinates
  if (amr_mesh.getConnectivity().size() == 0) {
    amr_mesh.computeConnectivity();
  }

  std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

  if (nameSuffix.size() > 0)
    outputPrefix = outputPrefix + nameSuffix;

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
  out << "      <CellData>\n";

  // write data array scalar fields in ascii
  for ( auto iter : names2index) {

    // get variables string name
    const std::string varName = iter.first;
    
    // get variable id
    int iVar = iter.second;    

    out << "      <DataArray type=\"Float64\" Name=\"" << varName << "\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl
	<< "          " << std::fixed;
    
    int ndata = amr_mesh.getConnectivity().size();
    for(int i = 0; i < ndata; i++)
      {
	// for now, only save ID (density)
	out << std::setprecision(6) << datah(i,fm[iVar]) << " ";
	if((i+1)%4==0 and i!=ndata-1)
	  out << std::endl << "          ";
      }
    out << std::endl << "        </DataArray>" << std::endl;
    
  } // end for iter
  
  out << "      </CellData>" << std::endl
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
	 << "    </PPointData>" << std::endl;

    pout << "    <PCellData>" << std::endl;

    for ( auto iter : names2index) {
      
      // get variables string name
      const std::string varName = iter.first;
      
      pout << "      <PDataArray type=\"Float64\" Name=\"" << varName<< "\" NumberOfComponents=\"1\"/>" << std::endl;
    } // end for iter
    
    pout << "    </PCellData>" << std::endl;

    pout << "    <PPoints>" << std::endl
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
  if (amr_mesh.isCommSet()) {
    // TODO - refactorx
    //MPI_Barrier(m_comm);
  }
#endif
  
} // writeVTK

// =======================================================
// =======================================================
void writeTest(AMRmesh               &amr_mesh,
	       std::string            filenameSuffix,
	       Kokkos::View<double*>  data)
{

  // copy data from device to host
  Kokkos::View<double*>::HostMirror datah = Kokkos::create_mirror(data);
  // copy device data to host
  Kokkos::deep_copy(datah, data);

  // dimension : 2 or 3 ?
  uint8_t dim = amr_mesh.getDim();
  
  // if not done already, compute mesh connectivity
  // and store nodes coordinates
  if (amr_mesh.getConnectivity().size() == 0) {
    amr_mesh.computeConnectivity();
  }

  std::string outputPrefix = "./data";
  
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
  out << "      <CellData>\n";

  // write data array scalar fields in ascii
  {
    
    out << "      <DataArray type=\"Float64\" Name=\"" << "data" << "\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl
	<< "          " << std::fixed;
    
    int ndata = amr_mesh.getConnectivity().size();
    for(int i = 0; i < ndata; i++)
      {
	// for now, only save ID (density)
	out << std::setprecision(6) << datah(i) << " ";
	if((i+1)%4==0 and i!=ndata-1)
	  out << std::endl << "          ";
      }
    out << std::endl << "        </DataArray>" << std::endl;
    
  }
  
  out << "      </CellData>" << std::endl
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
	 << "    </PPointData>" << std::endl;

    pout << "    <PCellData>" << std::endl;

    {
      
      // get variables string name
      const std::string varName = "data";
      
      pout << "      <PDataArray type=\"Float64\" Name=\"" << varName<< "\" NumberOfComponents=\"1\"/>" << std::endl;
    } // end for iter
    
    pout << "    </PCellData>" << std::endl;

    pout << "    <PPoints>" << std::endl
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
  if (amr_mesh.isCommSet()) {
    // TODO - refactorx
    //MPI_Barrier(m_comm);
  }
#endif

} // writeTest

// =======================================================
// =======================================================
void writeTest(AMRmesh             &amr_mesh,
	       std::string          filenameSuffix,
	       std::vector<double>  data)
{

  // dimension : 2 or 3 ?
  uint8_t dim = amr_mesh.getDim();
  
  // if not done already, compute mesh connectivity
  // and store nodes coordinates
  if (amr_mesh.getConnectivity().size() == 0) {
    amr_mesh.computeConnectivity();
  }

  std::string outputPrefix = "./data";
  
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
  out << "      <CellData>\n";

  // write data array scalar fields in ascii
  {
    
    out << "      <DataArray type=\"Float64\" Name=\"" << "data" << "\" NumberOfComponents=\"1\" format=\"ascii\">" << std::endl
	<< "          " << std::fixed;
    
    int ndata = amr_mesh.getConnectivity().size();
    for(int i = 0; i < ndata; i++)
      {
	// for now, only save ID (density)
	out << std::setprecision(6) << data[i] << " ";
	if((i+1)%4==0 and i!=ndata-1)
	  out << std::endl << "          ";
      }
    out << std::endl << "        </DataArray>" << std::endl;
    
  }
  
  out << "      </CellData>" << std::endl
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
	 << "    </PPointData>" << std::endl;

    pout << "    <PCellData>" << std::endl;

    {
      
      // get variables string name
      const std::string varName = "data";
      
      pout << "      <PDataArray type=\"Float64\" Name=\"" << varName<< "\" NumberOfComponents=\"1\"/>" << std::endl;
    } // end for iter
    
    pout << "    </PCellData>" << std::endl;

    pout << "    <PPoints>" << std::endl
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
  if (amr_mesh.isCommSet()) {
    // TODO - refactorx
    //MPI_Barrier(m_comm);
  }
#endif

} // writeTest - std::vector<double>

} // namespace euler_pablo

