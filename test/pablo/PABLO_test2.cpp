#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_PABLO.hpp"

using namespace bitpit;

// ======================================================================== //
/**
 *  \example PABLO_test2.cpp
 *
 * No user data involved here, just playing with AMR mesh structure:
 * - learn how to instanciate an mesh, cell ordererd with Morton indexing
 * - learn how findNeighbours works, with/without periodic borders 
 */
// ======================================================================== //

/**
 * This is a 3D only example. Just playing with neighbors through an edge.
 *
 * For example, look at cellId=7, all hanging edge doesn't have a neighbor.
 */
void run()
{

  //int nProcs;
  int rank;
#if BITPIT_ENABLE_MPI==1
  //MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  //nProcs = 1;
  rank   = 0;
#endif

  int dim = 3;

  int niter = 2;

  /**<Instantation of a nDimensional pablo uniform object.*/
  PabloUniform amr_mesh(dim);

  /**<set periodic border condition */
  amr_mesh.setPeriodic(0);
  amr_mesh.setPeriodic(1);
  amr_mesh.setPeriodic(2);
  amr_mesh.setPeriodic(3);
  amr_mesh.setPeriodic(4);
  amr_mesh.setPeriodic(5);

  
  /**<Refine globally four level and write the octree.*/
  for (int iter=1; iter<niter; ++iter){

    amr_mesh.adaptGlobalRefine();
    uint32_t nocts = amr_mesh.getNumOctants();

    printf("==============================================\n");
    printf("initial global refine iter %d (nbOcts=%d)\n",iter,nocts);
    printf("==============================================\n");


  } // end for iter

  printf("============================================================\n");
  printf("============================================================\n");

  /**< test refinement. Mark cell 0 for refinement */
  uint32_t cellId = 0;
  uint8_t do_refine=1;
  amr_mesh.setMarker(cellId,do_refine);
  amr_mesh.adapt();

#if BITPIT_ENABLE_MPI==1
  /**<(Load)Balance the octree over the processes.*/
  amr_mesh.loadBalance();
#endif

  amr_mesh.updateConnectivity();

  /*
   * print again mesh connectivty information
   */
  {
    // print neighbors id
    vector<uint32_t> neigh_t;
    vector<bool> isghost_t;

    uint32_t nocts = amr_mesh.getNumOctants();
    for (uint32_t i=0; i<nocts; ++i) {

      // print cell nodes location
      vector<array<double,3> > nodes = amr_mesh.getNodes(i);
      printf("rank %d octant %d (Morton = %lu): %f %f %f | %f %f %f | %f %f %f | %f %f %f\n           %f %f %f | %f %f %f | %f %f %f | %f %f %f\n",
             rank, i, amr_mesh.getMorton(i),
             nodes[0][0],nodes[0][1],nodes[0][2],
             nodes[1][0],nodes[1][1],nodes[1][2],
             nodes[2][0],nodes[2][1],nodes[2][2],
             nodes[3][0],nodes[3][1],nodes[3][2],
             nodes[4][0],nodes[4][1],nodes[4][2],
             nodes[5][0],nodes[5][1],nodes[5][2],
             nodes[6][0],nodes[6][1],nodes[6][2],
             nodes[7][0],nodes[7][1],nodes[7][2]);
          
      /*
       * codim=1 ==> faces
       */
      {
        int codim = 1;
        
        // number of faces per cell
        uint8_t nfaces = 2*dim;

        // get neighbors octant id face per face
        for (uint8_t iface=0; iface<nfaces; ++iface){
          amr_mesh.findNeighbours(i,iface,codim,neigh_t,isghost_t);
          printf("neighbors of %d through face %d are : ",i,iface);
          for (size_t ineigh=0; ineigh<neigh_t.size(); ++ineigh) {
            printf(" %d ",neigh_t[ineigh]);
          }
          printf("\n");
        } // end for iface

      } // end codim = 1

      
      /*
       * codim=2 ==> edges in 3D or corners in 2D
       */
      {
        int codim = 2;
        
        if (dim==2) {
          // Be careful, if findNeighboours through a corner is empty, there
          // are 2 possibilities:
          // - corner is actually touching external border
          // - one of the 2 faces (touching the corner) have a larger neighbor, i.e.
          //   the corner is actually a hanging node (a la p4est)

          // number of corners per cell
          uint8_t ncorner = 4; // number of corners in 2D

          for (uint8_t icorner=0; icorner<ncorner; ++icorner) {
            amr_mesh.findNeighbours(i,icorner,codim,neigh_t,isghost_t);
            printf("neighbors of %d through corner %d are : ",i,icorner);
            for (size_t ineigh=0; ineigh<neigh_t.size(); ++ineigh) {
              printf(" %d ",neigh_t[ineigh]);
            }
            printf("\n");
          }
            
        }

        if (dim==3) {
          uint8_t nedge = 12; // number of edges in 3D
          for (uint8_t iedge=0; iedge<nedge; ++iedge) {
            amr_mesh.findNeighbours(i,iedge,codim,neigh_t,isghost_t);
            printf("neighbors of %d through edge %d are : ",i,iedge);
            printf("|| number=%ld || ",neigh_t.size());
            for (size_t ineigh=0; ineigh<neigh_t.size(); ++ineigh) {
              printf(" %d ",neigh_t[ineigh]);
            }
            printf("\n");
          }
        }

      } // end codim = 2

      /*
       * codim=3 ==> only available in 3D, corners
       */
      if (dim==3) {
        int codim = 3;

        uint8_t ncorner = 8;
        for (uint8_t icorner=0; icorner<ncorner; ++icorner) {
          amr_mesh.findNeighbours(i,icorner,codim,neigh_t,isghost_t);
          printf("neighbors of %d through corner %d are : ",i,icorner);
          for (size_t ineigh=0; ineigh<neigh_t.size(); ++ineigh) {
            printf(" %d ",neigh_t[ineigh]);
          }
          printf("\n");
        }

      } // end codim = 3

    } // end for i
  }
  
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
