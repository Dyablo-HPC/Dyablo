/**
 *  \example mandelbrot.cpp
 */
#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_PABLO.hpp"

using namespace bitpit;

// maximum number of iterations
static const int NMAX=100;

// physical domain extent
static const double xmin = -2.25;
static const double xmax =  1.25;
static const double ymin = -1.50;
static const double ymax =  1.50;

static const double deltaX = xmax-xmin;
static const double deltaY = ymax-ymin;

/// rescale x coordinate from unit square to physical domain
double scaleX(double x) {
  return xmin + deltaX * x;
}

/// rescale y coordinate from unit square to physical domain
double scaleY(double y) {
  return ymin + deltaY * y;
}

/// refinement threashold
static const double epsilon = 0.05;

// ========================================================================
// ========================================================================
/**
 * Compute number of iterations of \f$ z_{n+1} = z_n^2 + c \f$ starting with
 * \f$ z_0=c\f$ to reach 4 in modulus.
 */
double
compute_nb_iters (double cx, double cy)
{
  // init number of iterations
  int j = 0;

  double norm = (cx*cx + cy*cy);
  double zx = cx;
  double zy = cy;
  double tmp;

  while (j <= NMAX and norm < 4) {
    tmp  = (zx*zx) - (zy*zy) + cx; // Real part 
    zy   = (2.*zx*zy) + cy;        // Imag part
    zx   = tmp;
    j++;
    norm = (zx*zx + zy*zy);
  }
  
  return (double) j;
  
} // compute_nb_iters

// ========================================================================
// ========================================================================
/**
 * Compute and save Mandelbrot set to file
 */
void compute_and_save_mandelbrot(PabloUniform& amr_mesh, size_t iter)
{
  uint32_t nocts = amr_mesh.getNumOctants();
  vector<double> oct_data(nocts, 0.0);
  vector<double> oct_data_level(nocts, 0.0);

  for (size_t i=0; i<nocts; ++i) {
    // get cell center coordinate in the unit domain
    std::array<double,3> center = amr_mesh.getCenter(i);

    // change into coordinates in the physical domain
    double x = scaleX(center[0]);
    double y = scaleY(center[1]);

    // compute pixel status, how many iterations for the Mandelbrot
    // series to diverge
    oct_data[i]       = compute_nb_iters(x,y);
    oct_data_level[i] = amr_mesh.getLevel(i);
  }
  
  amr_mesh.writeTest("mandelbrot_iter"+to_string(static_cast<unsigned long long>(iter)), oct_data);
  amr_mesh.writeTest("mandelbrot_level"+to_string(static_cast<unsigned long long>(iter)), oct_data_level);

#if BITPIT_ENABLE_MPI==1
  // save MPI rank
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    for (size_t i=0; i<nocts; ++i) {
      oct_data[i]       = rank;
    }

    amr_mesh.writeTest("mandelbrot_rank"+to_string(static_cast<unsigned long long>(iter)), oct_data);
    
  }
#endif
  
} // compute_and_save_mandelbrot

// ========================================================================
// ========================================================================
/**
 * Run the example.
 */
void run()
{

  /**<Instantation of a 2D pablo uniform object.*/
  PabloUniform amr_mesh(2);

  // start with a 32x32 array
  for (int i=0; i<5; ++i)
    amr_mesh.adaptGlobalRefine();
  
#if BITPIT_ENABLE_MPI==1
  // (Load)Balance the octree over the processes.
  amr_mesh.loadBalance();
#endif
  
  amr_mesh.updateConnectivity();  

  compute_and_save_mandelbrot(amr_mesh, 0);  

  // add several levels of refinement
  for (size_t iter=1; iter<=5; ++iter) {

    // refine all cells
    uint32_t nocts = amr_mesh.getNumOctants();

    for (size_t i=0; i<nocts; ++i){

      // get cell level
      uint8_t level = amr_mesh.getLevel(i);

      // only look at level 5 + iter - 1
      if (level == (5+iter-1)) {

	double x,y;
	double nbiter_center;
	
	// get cell center
	std::array<double,3> center = amr_mesh.getCenter(i);
	
	// change into coordinates in the physical domain
	x = scaleX(center[0]);
	y = scaleY(center[1]);
	
	// compute how many iterations for the Mandelbrot
	// series to diverge at center
	nbiter_center = compute_nb_iters(x,y);
	
	// get cell corner nodes
	vector<array<double,3> > nodes = amr_mesh.getNodes(i);

	double interpol = 0.0;
	for (int ic=0; ic<4; ++ic) {
	  x = scaleX(nodes[ic][0]);
	  y = scaleY(nodes[ic][1]);
	  interpol += compute_nb_iters(x,y);
	}
	interpol /= 4.0;
	interpol = fabs(interpol);
	
	bool should_refine = false;
	// compute how much linear interpolation is different from
	// value at cell center and
	// decide if cell need to be flagged for refinement
	if (interpol > 0) {
	  if (nbiter_center / interpol > 1+epsilon or
	      nbiter_center / interpol < 1-epsilon)
	    should_refine = true;
	}

	if (should_refine) {
	  //printf("refine %d\n",i);
	  amr_mesh.setMarker(i, 1);
	}
	
      } // end if level
      
    } // end if nocts
    
    amr_mesh.adapt();

#if BITPIT_ENABLE_MPI==1
    /**<(Load)Balance the octree over the processes.*/
    amr_mesh.loadBalance();
#endif

    amr_mesh.updateConnectivity();  
    
    compute_and_save_mandelbrot(amr_mesh, iter);

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

  int nProcs = 1;
  int rank = 0;
#if BITPIT_ENABLE_MPI==1
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
