#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_PABLO.hpp"

using namespace std;
using namespace bitpit;

// ======================================================================== //
/**
 *  \example PABLO_test1.cpp
 */
// ======================================================================== //

/**
 * Run the example.
 *
 * in 2D, after 3 global refine iteration, the mesh look like:
 *
 * 10  11  14  15
 * 
 *  8   9  12  13
 *
 *  2   3   6   7
 *
 *  0   1   4   5
 *
 * If we decide to refine cell 3, then the mesh looks like:
 *
 * 13    14     17     18
 *
 *
 * 11    12     15     16
 *
 *
 *  2   5 6     9      10
 *      3 4
 *
 *  0     1     7       8
 * 
 * so you can cross-check the mesh connectivity.
 */
void run(int dim)
{
  int iter = 0;

  /**<Instantation of a nDimensional pablo uniform object.*/
  PabloUniform amr_mesh(dim);

  /**<Refine globally four level and write the octree.*/
  for (iter=1; iter<3; iter++){

    printf("===================================\n");
    printf("initial global refine iter %d\n",iter);
    printf("===================================\n");

    amr_mesh.adaptGlobalRefine();
    uint32_t nocts = amr_mesh.getNumOctants();

    vector<uint32_t> neigh, neigh_t;
    vector<bool> isghost, isghost_t;

    for (unsigned int i=0; i<nocts; i++){
      // print cell nodes location
      vector<array<double,3> > nodes = amr_mesh.getNodes(i);

      if (dim==2) {
	printf("nodes %d : %f %f | %f %f | %f %f | %f %f \n",i,
	       nodes[0][0],nodes[0][1],
	       nodes[1][0],nodes[1][1],
	       nodes[2][0],nodes[2][1],
	       nodes[3][0],nodes[3][1]);
      } else {
	printf("nodes %d : %f %f %f | %f %f %f | %f %f %f | %f %f %f\n           %f %f %f | %f %f %f | %f %f %f | %f %f %f\n",i,
	       nodes[0][0],nodes[0][1],nodes[0][2],
	       nodes[1][0],nodes[1][1],nodes[1][2],
	       nodes[2][0],nodes[2][1],nodes[2][2],
	       nodes[3][0],nodes[3][1],nodes[3][2],
	       nodes[4][0],nodes[4][1],nodes[4][2],
	       nodes[5][0],nodes[5][1],nodes[5][2],
	       nodes[6][0],nodes[6][1],nodes[6][2],
	       nodes[7][0],nodes[7][1],nodes[7][2]);
      }
	
      // print neighbors id per face
      // if neigh_t is empty, it means there is no "genuine" neighbor
      // and thus the given octant touches the external border.
      // In a real application we should the border condition.
      neigh.clear();
      isghost.clear();

      int codim = 1;
      uint8_t nfaces = 2*dim;
      for (uint8_t iface=0; iface<nfaces; iface++){
	amr_mesh.findNeighbours(i,iface,codim,neigh_t,isghost_t);
	printf("neighbors of %d through face %d are : ",i,iface);
	for (int ineigh=0; ineigh<neigh_t.size(); ++ineigh) {
	  printf(" %d ",neigh_t[ineigh]);
	}
	printf("\n");
	
      } // end for iface
      
    } // end for nocts
  } // end for iter

  printf("============================================================\n");
  printf("============================================================\n");

  /**< test refinement. Mark cell 3 for refinement */
  amr_mesh.setMarker(3,1);
  amr_mesh.adapt();
  amr_mesh.updateConnectivity();

  // print again mesh connectivty information
  {
    // print neighbors id
    vector<uint32_t> neigh_t;
    vector<bool> isghost_t;

    uint32_t nocts = amr_mesh.getNumOctants();
    for (unsigned int i=0; i<nocts; i++) {

      // print cell nodes location
      vector<array<double,3> > nodes = amr_mesh.getNodes(i);
      if (dim==2) {
	printf("nodes %d : %f %f | %f %f | %f %f | %f %f \n",i,
	       nodes[0][0],nodes[0][1],
	       nodes[1][0],nodes[1][1],
	       nodes[2][0],nodes[2][1],
	       nodes[3][0],nodes[3][1]);
      } else {
	printf("nodes %d : %f %f %f | %f %f %f | %f %f %f | %f %f %f\n           %f %f %f | %f %f %f | %f %f %f | %f %f %f\n",i,
	       nodes[0][0],nodes[0][1],nodes[0][2],
	       nodes[1][0],nodes[1][1],nodes[1][2],
	       nodes[2][0],nodes[2][1],nodes[2][2],
	       nodes[3][0],nodes[3][1],nodes[3][2],
	       nodes[4][0],nodes[4][1],nodes[4][2],
	       nodes[5][0],nodes[5][1],nodes[5][2],
	       nodes[6][0],nodes[6][1],nodes[6][2],
	       nodes[7][0],nodes[7][1],nodes[7][2]);
      }
      
      int codim = 1;
      uint8_t nfaces = 2*dim;
      for (uint8_t iface=0; iface<nfaces; iface++){
	amr_mesh.findNeighbours(i,iface,codim,neigh_t,isghost_t);
	printf("neighbors of %d through face %d are : ",i,iface);
	for (int ineigh=0; ineigh<neigh_t.size(); ++ineigh) {
	  printf(" %d ",neigh_t[ineigh]);
	}
	printf("\n");
      } // end for iface
    } // end for i
  }

  
  /**<Define a center point and a radius.*/
  double xc, yc, zc;
  xc = yc = zc = 0.5;
  double radius = 0.25;

  /**<Define vectors of data.*/
  uint32_t nocts = amr_mesh.getNumOctants();
  vector<double> oct_data(nocts, 0.0);

  /**<Assign a data to the octants with at least one node inside the circle.*/
  for (unsigned int i=0; i<nocts; i++){
    /**<Compute the nodes of the octant.*/
    vector<array<double,3> > nodes = amr_mesh.getNodes(i);

    if (dim==2) {
      for (int j=0; j<4; j++){
	double x = nodes[j][0];
	double y = nodes[j][1];
	if ((pow((x-xc),2.0)+pow((y-yc),2.0) <= pow(radius,2.0))){
	  oct_data[i] = 1.0;
	}
      }
    } else {
      for (int j=0; j<8; j++){
	double x = nodes[j][0];
	double y = nodes[j][1];
	double z = nodes[j][2];
	if ((pow((x-xc),2.0)+pow((y-yc),2.0)+pow((z-zc),2.0) <= pow(radius,2.0))){
	  oct_data[i] = 1.0;
	}
      }
    }
  }

  /**<Update the connectivity and write the octree.*/
  iter = 0;
  amr_mesh.updateConnectivity();
  amr_mesh.writeTest("PABLO_test1_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data);

  /**<Smoothing iterations on initial data*/
  int start = 1;
  for (iter=start; iter<start+3; iter++){
    vector<double> oct_data_smooth(nocts, 0.0);
    vector<uint32_t> neigh, neigh_t;
    vector<bool> isghost, isghost_t;
    uint8_t iface, nfaces;
    int codim;
    for (unsigned int i=0; i<nocts; i++){
      neigh.clear();
      isghost.clear();

      /**<Find neighbours through faces (codim=1) and edges (codim=2) of the octants*/
      for (codim=1; codim<3; codim++){
	if (codim == 1){
	  nfaces = 2*dim;
	}
	else if (codim == 2){
	  nfaces = dim==2 ? 4 : 12;
	}
	for (iface=0; iface<nfaces; iface++){
	  amr_mesh.findNeighbours(i,iface,codim,neigh_t,isghost_t);
	  neigh.insert(neigh.end(), neigh_t.begin(), neigh_t.end());
	  isghost.insert(isghost.end(), isghost_t.begin(), isghost_t.end());
	}
      }

      /**<Smoothing data with the average over the one ring neighbours of octants*/
      oct_data_smooth[i] = oct_data[i]/(neigh.size()+1);
      for (unsigned int j=0; j<neigh.size(); j++){
	if (isghost[j]){
	  /**< Do nothing - No ghosts: is a serial test.*/
	}
	else{
	  oct_data_smooth[i] += oct_data[neigh[j]]/(neigh.size()+1);
	}
      }
    }

    /**<Update the connectivity and write the octree.*/
    amr_mesh.updateConnectivity();
    amr_mesh.writeTest("PABLO_test11_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data_smooth);

    oct_data = oct_data_smooth;
    
  } // end for iter
  
} // end run

/*!
 * Main program.
 */
int main(int argc, char *argv[])
{
#if BITPIT_ENABLE_MPI==1
  MPI_Init(&argc,&argv);
#else
  BITPIT_UNUSED(argc);
  BITPIT_UNUSED(argv);
#endif

  int nProcs;
  int rank;
#if BITPIT_ENABLE_MPI==1
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  nProcs = 1;
  rank   = 0;
#endif

  // Initialize the logger
  log::manager().initialize(log::SEPARATE, false, nProcs, rank);
  log::cout() << fileVerbosity(log::NORMAL);
  log::cout() << consoleVerbosity(log::QUIET);

  int dim = 2;
  if (argc>1) {
    dim = atoi(argv[1]);
    if (dim != 2 and dim != 3)
      dim=2;
  }
  
  // Run the example
  try {
    run(dim);
  } catch (const std::exception &exception) {
    log::cout() << exception.what();
    exit(1);
  }

#if BITPIT_ENABLE_MPI==1
  MPI_Finalize();
#endif
}
