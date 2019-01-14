/**
 *  \example mandelbrot.cpp
 */
#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_PABLO.hpp"

using namespace std;
using namespace bitpit;


static const int NMAX=100;

static const double xmin = -2.25;
static const double xmax =  1.25;
static const double ymin = -1.50;
static const double ymax =  1.50;

static const double deltaX = xmax-xmin;
static const double deltaY = ymax-ymin;

double scaleX(double x) {
  return xmin + deltaX * x;
}

double scaleY(double y) {
  return ymin + deltaY * y;
}

// ========================================================================
// ========================================================================
/**
 * Compute number of iterations of \f$ z_{n+1} = z_n^2 + c \f$ starting with
 * \f$ z_0=c\f$ to reach 4 in modulus.
 */
double
compute_nb_iters (double cx, double cy)
{
  int j = 0;
  double norm = (cx*cx + cy*cy);
  double zx = cx;
  double zy = cy;
  double tmp;

  while (j <= NMAX and norm < 4) {
    tmp  = (zx*zx) - (zy*zy) + cx;// Real part 
    zy   = (2.*zx*zy) + cy; //Imag part
    zx   = tmp;
    j++;
    norm = (zx*zx + zy*zy);
  }
  
  return (double) j;
  
} // compute_nb_iters


// ========================================================================
// ========================================================================
/**
 * Run the example.
 */
void run()
{

  int nProcs;
  int rank;
#if BITPIT_ENABLE_MPI==1
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  nProcs = 1;
  rank   = 0;
#endif

  /**<Instantation of a 2D pablo uniform object.*/
  PabloUniform amr_mesh(2);

  // start with a 32x32 array
  for (int i=0; i<5; ++i)
    amr_mesh.adaptGlobalRefine();

  for (int iter=0; iter<2; ++iter) {

    // refine all cells
    uint32_t nocts = amr_mesh.getNumOctants();

    for (unsigned int i=0; i<nocts; i++){
      // print cell center location
      std::array<double,3> center = amr_mesh.getCenter(i);
      //printf("center %d : %f %f \n",i,
      //	     scaleX(center[0]),scaleY(center[1]));

      // decide if cell need to be flagged for refinement
      
    }

    amr_mesh.adapt();

#if BITPIT_ENABLE_MPI==1
    /**<(Load)Balance the octree over the processes.*/
    amr_mesh.loadBalance();
#endif

    amr_mesh.updateConnectivity();  
    
    /**<Define vectors of data.*/
    uint32_t nocts2 = amr_mesh.getNumOctants();
    vector<double> oct_data(nocts2, 0.0);
    
    /**<Assign a data to the octants with at least one node inside the circle.*/
    for (unsigned int i=0; i<nocts2; i++){
      oct_data[i] = rank;
    }
    
    amr_mesh.updateConnectivity();
    amr_mesh.writeTest("PABLO_test0_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data);

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
