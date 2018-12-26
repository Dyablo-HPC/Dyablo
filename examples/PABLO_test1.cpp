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
 */
void run()
{
  int iter = 0;

  /**<Instantation of a 2D pablo uniform object.*/
  PabloUniform amr_mesh(2);

  /**<Refine globally four level and write the octree.*/
  for (iter=1; iter<3; iter++){
    amr_mesh.adaptGlobalRefine();
    uint32_t nocts = amr_mesh.getNumOctants();
    for (unsigned int i=0; i<nocts; i++){
      vector<array<double,3> > nodes = amr_mesh.getNodes(i);
      printf("nodes %d : %f %f | %f %f | %f %f | %f %f \n",i,
	     nodes[0][0],nodes[0][1],
	     nodes[1][0],nodes[1][1],
	     nodes[2][0],nodes[2][1],
	     nodes[3][0],nodes[3][1]);
    }  
  }

  printf("============================================================\n");
  
  /**<Define a center point and a radius.*/
  double xc, yc;
  xc = yc = 0.5;
  double radius = 0.25;

  /**<Define vectors of data.*/
  uint32_t nocts = amr_mesh.getNumOctants();
  vector<double> oct_data(nocts, 0.0);

  /**<Assign a data to the octants with at least one node inside the circle.*/
  for (unsigned int i=0; i<nocts; i++){
    /**<Compute the nodes of the octant.*/
    vector<array<double,3> > nodes = amr_mesh.getNodes(i);
    for (int j=0; j<4; j++){
      double x = nodes[j][0];
      double y = nodes[j][1];
      if ((pow((x-xc),2.0)+pow((y-yc),2.0) <= pow(radius,2.0))){
	oct_data[i] = 1.0;
      }
    }
  }

  /**<Update the connectivity and write the octree.*/
  iter = 0;
  amr_mesh.updateConnectivity();
  amr_mesh.writeTest("test1_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data);

  /**<Smoothing iterations on initial data*/
  int start = 1;
  for (iter=start; iter<start+25; iter++){
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
	  nfaces = 4;
	}
	else if (codim == 2){
	  nfaces = 4;
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
    amr_mesh.writeTest("test1_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data_smooth);

    oct_data = oct_data_smooth;
  }
}

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

  // Run the example
  try {
    run();
  } catch (const std::exception &exception) {
    log::cout() << exception.what();
    exit(1);
  }

#if BITPIT_ENABLE_MPI==1
  MPI_Finalize();
#endif
}
