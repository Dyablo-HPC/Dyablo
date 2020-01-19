/**
 *  \example mandelbrot_kokkos.cpp
 *  \author Pierre Kestener
 *
 * Compute Mandelbrot set in an iterative way, starting with a coarse
 * uniform grid, then apply N times a refinement operator, and saving
 * the results (VTU unstructured grid file format) for each intermediate
 * step.
 *
 * Same as mandelbrot.cpp but computation kernel is done with Kokkos.
 * For now, only Kokkos/OpenMP backend can be used.  
 */
#include <memory> // for shared pointer


#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_PABLO.hpp"
#include "shared/kokkos_shared.h"
//#include "shared/bitpit_common.h" // for AMRmesh type
using AMRmesh = bitpit::PabloUniform;

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
KOKKOS_INLINE_FUNCTION
double scaleX(double x) {
  return xmin + deltaX * x;
}

/// rescale y coordinate from unit square to physical domain
KOKKOS_INLINE_FUNCTION
double scaleY(double y) {
  return ymin + deltaY * y;
}

/// refinement threshold
static constexpr double epsilon = 0.05;

// ========================================================================
// ========================================================================
/**
 * Compute number of iterations of complex number sequence defined by
 * \f$ z_{n+1} = z_n^2 + c \f$
 * starting with \f$ z_0=c\f$ and stopping when \f$ |z^2| \f$ reaches 4 or
 * after maximum iterations NMAX.
 *
 * \param[in] cx x coordinate of c
 * \param[in] cy y coordinate of c
 */
KOKKOS_INLINE_FUNCTION
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

// ==========================================================
// ==========================================================
class MandelbrotRefine {

public:
  MandelbrotRefine(std::shared_ptr<AMRmesh> pmesh, int iter) :
    pmesh(pmesh),
    iter(iter) {};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    int iter)
  {
    
    // iterate functor for refinement
    MandelbrotRefine functor(pmesh, iter);
    Kokkos::parallel_for(pmesh->getNumOctants(), functor);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& i) const
  {

    // get cell level
    uint8_t level = pmesh->getLevel(i);
    
    // only look at level 5 + iter - 1
    if (level == (5+iter-1)) {
      
      double x,y;
      double nbiter_center;
      
      // get cell center
      std::array<double,3> center = pmesh->getCenter(i);
      
      // change into coordinates in the physical domain
      x = scaleX(center[0]);
      y = scaleY(center[1]);
      
      // compute how many iterations for the Mandelbrot
      // series to diverge at center
      nbiter_center = compute_nb_iters(x,y);
      
      // get cell corner nodes
      std::vector<std::array<double,3> > nodes = pmesh->getNodes(i);
      
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
	pmesh->setMarker(i, 1);
      }
      
    } // end if level    
    
  } // operator()
  
  std::shared_ptr<AMRmesh> pmesh;
  int iter;
  
}; // MandelbrotRefine

// ========================================================================
// ========================================================================
/**
 * Compute and save Mandelbrot set to file.
 *
 * \param[in] amr_mesh reference to PabloUniform object
 * \param[in] iter number used to suffix output file name
 *
 */
void compute_and_save_mandelbrot(std::shared_ptr<AMRmesh> pmesh, 
                                 size_t iter)
{
  uint32_t nocts = pmesh->getNumOctants();

  Kokkos::View<double*, dyablo::Device> oct_data("oct_data",nocts);
  Kokkos::View<double*, dyablo::Device> oct_data_level("oct_data_level",nocts);

  Kokkos::parallel_for(nocts, KOKKOS_LAMBDA(const size_t &i) {
      
      // get cell center coordinate in the unit domain
      std::array<double,3> center = pmesh->getCenter(i);
      
      // change into coordinates in the physical domain
      double x = scaleX(center[0]);
      double y = scaleY(center[1]);
      
      // compute pixel status, how many iterations for the Mandelbrot
      // series to diverge
      oct_data(i)       = compute_nb_iters(x,y);
      oct_data_level(i) = pmesh->getLevel(i);
      
    });


  // FIXME : make writeTest accept a DataArray, or
  // convert DataArray to std::vector before calling writeTest
  std::vector<double> oct_data_v(nocts, 0.0);
  std::vector<double> oct_data_level_v(nocts, 0.0);

  for (size_t i=0; i<nocts; ++i) {
    oct_data_v[i]       = oct_data(i);
    oct_data_level_v[i] = oct_data_level(i);
  }

  pmesh->writeTest("mandelbrot_iter"+std::to_string(static_cast<unsigned long long>(iter)), oct_data_v);
  pmesh->writeTest("mandelbrot_level"+std::to_string(static_cast<unsigned long long>(iter)), oct_data_level_v);

#if BITPIT_ENABLE_MPI==1
  // save MPI rank
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    for (size_t i=0; i<nocts; ++i) {
      oct_data_v[i]       = rank;
    }

    pmesh->writeTest("mandelbrot_rank"+std::to_string(static_cast<unsigned long long>(iter)), oct_data_v);
    
  }
#endif
  
} // compute_and_save_mandelbrot

// ========================================================================
// ========================================================================
/**
 * Main driver for Mandelbrot set computation.
 *
 * Here are the computing steps:
 * 0. create a uniformly refine coarse mesh (typically 32 by 32), 
 *    then compute and save the coarse regular grid Mandelbrot set.
 * 1. Repeat 5 times:
 *    - flags cell for refinement when Mandelbrot set requires smaller
 *      cell for better evaluation
 *    - apply mesh refinement
 *    - compute and save the new refined Mandelbrot set.
 */
void run()
{

  /**<Instantation of a 2D pablo uniform object.*/
  std::shared_ptr<AMRmesh> pmesh = std::make_shared<AMRmesh>(2);

  // start with a 32x32 array
  for (int i=0; i<5; ++i)
    pmesh->adaptGlobalRefine();
  
#if BITPIT_ENABLE_MPI==1
  // (Load)Balance the octree over the processes.
  pmesh->loadBalance();
#endif
  
  pmesh->updateConnectivity();  

  compute_and_save_mandelbrot(pmesh, 0);  

  // add several levels of refinement
  for (size_t iter=1; iter<=5; ++iter) {

    // refine all cells
    MandelbrotRefine::apply(pmesh, iter);
    
    pmesh->adapt();

#if BITPIT_ENABLE_MPI==1
    /**<(Load)Balance the octree over the processes.*/
    pmesh->loadBalance();
#endif

    pmesh->updateConnectivity();  
    
    compute_and_save_mandelbrot(pmesh, iter);

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

  Kokkos::initialize(argc, argv);
  
  int rank=0;
  int nRanks=1;
  
  {
    std::cout << "##########################\n";
    std::cout << "KOKKOS CONFIG             \n";
    std::cout << "##########################\n";
    
    std::ostringstream msg;
    std::cout << "Kokkos configuration" << std::endl;
    if ( Kokkos::hwloc::available() ) {
      msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
          << "] x CORE["    << Kokkos::hwloc::get_available_cores_per_numa()
          << "] x HT["      << Kokkos::hwloc::get_available_threads_per_core()
          << "] )"
          << std::endl ;
    }
    Kokkos::print_configuration( msg );
    std::cout << msg.str();
    std::cout << "##########################\n";


#if BITPIT_ENABLE_MPI==1
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  }
  
  // Initialize the logger
  log::manager().initialize(log::SEPARATE, false, nRanks, rank);
  log::cout() << fileVerbosity(log::NORMAL);
  log::cout() << consoleVerbosity(log::QUIET);

  // Run the example
  try {
    run();
  } catch (const std::exception &exception) {
    log::cout() << exception.what();
    exit(1);
  }

  Kokkos::finalize();
  
#if BITPIT_ENABLE_MPI==1
  MPI_Finalize();
#endif
}
