/**
 * Compute the Mandelbrot set on an adaptive mesh.
 *
 * We do not try to seek for performance.
 * Adaptive mesh is stored using an Kokkos::UnorderedMap class where
 * inserted key,value pairs are made of:
 * - key is actually the Morton key (using type morton_key_t)
 * - value is a structure containing different fields.
 *
 * \sa test_unordered_map_io3.cpp
 *
 */
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <array>
#include <cstdint>
#include <vector>

#include "shared/real_type.h"
#include "shared/kokkos_shared.h"

#include "shared/morton_utils.h"
#include "amr_key.h"

#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_Sort.hpp>

// MPI support
#ifdef DYABLO_USE_MPI
#include "utils/mpi/GlobalMpiSession.h"
#endif // DYABLO_USE_MPI

// VTK IO implementation
#include "utils/io/VTKWriter.h" // for VTK_WRITE_ENUM class
#include "utils/io/IO_VTK_shared.h"


enum index_status_t {
  INDEX_UNINITIALIZED = -1
};

/**
 * Some parameters specific to Mandelbrot set computation
 * 
 */

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

/// refinement threashold
static constexpr double epsilon = 0.01;

// ========================================================================
// ========================================================================
/**
 * Compute number of iterations of \f$ z_{n+1} = z_n^2 + c \f$ starting with
 * \f$ z_0=c\f$ to reach 4 in modulus.
 */
KOKKOS_INLINE_FUNCTION
double
compute_nb_iters (double cx, double cy)
{
  // maximum number of iterations
  constexpr int NMAX=512;

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

namespace dyablo {

/** typedef Point holding coordinates of a point. */
template<int dim>
using Point = std::array<real_t, dim>;

bool VERBOSE = false;

/**
 * Minimalist metadata to stored in hash table for AMR.
 */
struct metadata_t {

  //! genuine Morton key (space location + tree encoding)
  morton_key_t key;

  //! address where to stored heavy data; optimally index ranges from 0
  //! to N-1, where N is the total number of keys, and index is "ordered"
  //! by the morton index (total order with space locality preserving)
  int64_t index;

  //! data
  double data;
  
  //! metadata used to stored cell status (uninitialized, To_be_removed, ...)
  CellStatus status;


  KOKKOS_INLINE_FUNCTION
  metadata_t() :
    key(), index(-1), data(0.0), status(CELL_INVALID) {}
  
  KOKKOS_INLINE_FUNCTION
  metadata_t(morton_key_t _key, int64_t _index, double _data, CellStatus _status) :
    key(_key), index(_index), data(_data), status(_status)
  {}
  
}; // struct metadata_t

/**
 * MandelbrotMap uses key = amr_key_t (morton + tree + level), this allows 
 * to have temporarily items with same location but different sizes
 * (i.e. different levels)
 * this happens e.g. while performing refine operation).
 */
using MandelbrotMap = Kokkos::UnorderedMap<amr_key_t, metadata_t, Device>;

// =========================================================================
// =========================================================================
/**
 * Functor to compute a Kokkos::UnorderedMap size (number of valid entries).
 *
 * Not really needed, since the functionnality already exists as 
 * UnorderedMap method named size().
 *
 */
struct ComputeMapSize
{
  
  using execution_space = typename MandelbrotMap::execution_space;

  MandelbrotMap mandelbrotMap;

  ComputeMapSize( MandelbrotMap mandelbrotMap ) :
    mandelbrotMap(mandelbrotMap)
  {}

  static uint32_t getSize(MandelbrotMap _mandelbrotMap)
  {
    uint32_t size = 0;
    ComputeMapSize functor(_mandelbrotMap);
    Kokkos::parallel_reduce(_mandelbrotMap.capacity(), functor, size);
    execution_space().fence();
    return size;
  }
  
  /*
   * 2D and 3D versions.
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const uint32_t& i, uint32_t& count) const
  {
    if (mandelbrotMap.valid_at(i))
      ++count;
    
  } // operator() - 2d/3d
  
  KOKKOS_INLINE_FUNCTION
  void init( uint32_t & update ) const { update = 0 ; }

  KOKKOS_INLINE_FUNCTION
  void join( volatile       uint32_t & update , 
	     volatile const uint32_t & input ) const { update += input ; }
  
}; // struct ComputeMapSize

// =========================================================================
// =========================================================================
/**
 * Functor to initialize a MandelbrotMap with
 * a uniform coarse mesh.
 *
 * kind of bootstrap.
 *
 * Remind that 
 *  - amr_key_t    encodes space + level + tree
 *  - morton_key_t encodes space + tree
 * Please note that they use very different encoding.
 *
 * TODO / Future version: update to take into account TreeId / connectivity.
 */
struct FillCoarseMap
{
  
  MandelbrotMap mandelbrotMap;
  int level_min, level_max;
  int N; // linear size : 2**level_min

  FillCoarseMap( MandelbrotMap mandelbrotMap, int level_min, int level_max ) :
    mandelbrotMap(mandelbrotMap),
    level_min(level_min),
    level_max(level_max),
    N(1<<level_min)
  {
    int size = N*N;
    Kokkos::parallel_for(size, *this);
  }

  /*
   * 2D version.
   */
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const
  {
    int iy = i/N;
    int ix = i-iy*N;

    // in unit of the smallest resolution (for genuine morton key)
    int mx = ix << (level_max-level_min);
    int my = iy << (level_max-level_min);

    const int treeId = 0;

    int64_t key1a = compute_morton_key(ix,iy);
    int64_t key1b = encode_level_tree(level_min, treeId);
    int64_t key2a = compute_morton_key(mx,my);
    int64_t key2b = encode_tree(0);
    
    amr_key_t     key(key1a, key1b);
    morton_key_t mkey(key2a, key2b);

    // compute Mandelbrot set "pixel value"
    real_t dx = 1.0/N;
    real_t dy = 1.0/N;
    real_t x = scaleX( (ix+0.5)*dx );
    real_t y = scaleY( (iy+0.5)*dy );
    double data = compute_nb_iters(x,y);
    
    metadata_t value(mkey, INDEX_UNINITIALIZED, data, CELL_REGULAR);
    
    mandelbrotMap.insert( key, value );
    
  } // operator()  
  
}; // struct FillCoarseMap

// =========================================================================
// =========================================================================
/**
 * Functor to refine a MandelbrotMap.
 *
 * the refinement criterium is adapted from the Mariani-Silver Algorithm, i.e.
 * for each each we compare the value at center with the average pixel of the
 * four sub-cell; if they agree up to epsilon, we do not refine; if they 
 * are strongly different, it means we need to refine (so new cells are 
 * inserted, and parent cell is removed).
 */
struct Refine
{
  
  enum refine_mode_t {
    INSERT_REFINED_CELLS,
    DELETE_OLD_CELLS,
    RESET_STATUS
  };

  /**
   * a Kokkos::UnorderedMap where
   * - key is morton+treeId
   * - value is aggregate key + level
   */
  MandelbrotMap mandelbrotMap;

  //! min and max refinment level
  int level_min, level_max;
  
  /**
   * refine mode: erase or insert; we can't erase and insert in an
   * Unordered map at the same time
   */
  refine_mode_t refine_mode;

  //! constructor
  Refine( MandelbrotMap mandelbrotMap,
	  int level_min,
	  int level_max,
	  refine_mode_t refine_mode) :
    mandelbrotMap(mandelbrotMap),
    level_min(level_min),
    level_max(level_max),
    refine_mode(refine_mode)
  {}

  /**
   * \param[in,out] _mandelbrotMap is the hash table which is use to perform refinement
   * \param[in] level_min
   * \param[in] level_max
   *
   */
  static void apply(MandelbrotMap _mandelbrotMap,
		    int level_min,
		    int level_max) {
    
    // step 0 : create functor
    Refine functor(_mandelbrotMap, level_min, level_max,
		   Refine::INSERT_REFINED_CELLS);

    // for each level
    int nbIter = level_max-level_min;
    for (int iter=0; iter<nbIter; ++iter) {

      // monitoring map size
      printf("[iter %d] Map size=%d (end)\n",iter,_mandelbrotMap.size());
      
      //
      // step1:
      //  - check if new refine cells need to be inserted,
      //  - insert them,
      //  - and mark old coarse cell for removal
      //
      functor.refine_mode = Refine::INSERT_REFINED_CELLS;
      
      // perform refinement
      Kokkos::parallel_for(_mandelbrotMap.capacity(), functor);
      
      if (_mandelbrotMap.failed_insert())
	printf("Something went wrong in refinement operation\n");
      
      //
      // step2: remove old coarse cells
      //
      
      // check erasable state - should be false (default)
      if (VERBOSE) std::cout << "is data map erasable ? " << _mandelbrotMap.erasable() << "\n";
      
      // _mandelbrotMap must be in erasable state
      _mandelbrotMap.begin_erase();
    
      // check erasable state again - should be true now
      if (VERBOSE) std::cout << "is data map erasable ? " << _mandelbrotMap.erasable() << "\n";
    
      // perform refinement
      functor.refine_mode = Refine::DELETE_OLD_CELLS;
      Kokkos::parallel_for(_mandelbrotMap.capacity(), functor);
      
      // actually perform the erase operations
      _mandelbrotMap.end_erase();
      
      // check erasable state again - should be back to false
      if (VERBOSE) std::cout << "is data map erasable ? " << _mandelbrotMap.erasable() << "\n";

      //
      // step3: reset cell status 
      //
      // VERY strange mandelbrotMap.size() gives the wrong result ?! To be analyzed
      //int map_size = _mandelbrotMap.size();
      int map_size = ComputeMapSize::getSize( _mandelbrotMap);
      printf("[iter %d] Map size=%d (end)\n",iter,map_size);
      functor.refine_mode = Refine::RESET_STATUS;
      Kokkos::parallel_for(_mandelbrotMap.capacity(), functor);

      Kokkos::fence();
      
    } // end for iter    
    
  } // apply
  
  
  KOKKOS_INLINE_FUNCTION
  void insert_refined_cells(const int& i) const
  {

    // retrieve key,value pair
    amr_key_t    key = mandelbrotMap.key_at(i);
    metadata_t value = mandelbrotMap.value_at(i);

    CellStatus status = value.status;

    /*
     * Avoid uninitialized cells (these are newly inserted cell)
     * and also for invalid cells (sometimes, don't know if it is a bug,
     * newly inserted cells have wrong data, i.e. status is zero
     * which is exactly CELL_INVALID !?)
     */
    if (status != CELL_UNINITIALIZED and status != CELL_INVALID) {
      
      uint8_t  level  = key.get_level();
      uint16_t treeId = key.get_treeId();
      
      uint64_t morton = key.get_morton();
      
      // decode morton index
      // ix and iy should lie in [0, 2**level_max-1]
      uint32_t ix = morton_extract_bits<2,IX>(morton);
      uint32_t iy = morton_extract_bits<2,IY>(morton);
      
      int N = 1<<level;
      real_t dx = 1.0/N;
      real_t dy = 1.0/N;
      
      // compute cell-center x,y coordinates in real space [0,1]^2
      real_t xc = (ix+0.5)*dx;
      real_t yc = (iy+0.5)*dy;
      real_t x = scaleX( xc );
      real_t y = scaleY( yc );

      // determine if refinement is needed
      // compute nb iter at cell center
      double datac = compute_nb_iters(x,y);

      // compute nb iter at the corners
      double data[4], average=0.0;
      for (int index=0; index<4; ++index) {

	int di = 2*(  index     & 0x1 )-1;
	int dj = 2*( (index>>1) & 0x1 )-1;
	x = scaleX( xc + di*0.25*dx );
	y = scaleY( yc + dj*0.25*dy );
	data[index] = compute_nb_iters(x,y);
	average += data[index];
      }
      average /= 4;

      bool refinement_needed = false;      
      if ( datac / average > 1+epsilon or
	   datac / average < 1-epsilon )
	refinement_needed = true;
      
      if ( refinement_needed ) {
	// refinement is need
	// create/insert new cells at fine level with metadata

	// build the new child (keys, values) and insert

	// if(level==5)
	//   printf("HOUSTON %d %d %ld -- %ld %d %d\n",i,value.status,value.index,
	// 	 value.key[0],
	// 	 morton_extract_bits<dim,IX>(value.key[0]),
	// 	 morton_extract_bits<dim,IY>(value.key[0]) );
	
	{
	  int ixx = ix<<1;
	  int iyy = iy<<1;
	  amr_key_t ckey(compute_morton_key(ixx,iyy), encode_level_tree(level+1,treeId));

	  int mx = ixx << (level_max-level-1);
	  int my = iyy << (level_max-level-1);
	  morton_key_t mkey(compute_morton_key(mx,my),encode_tree(treeId));
	  
	  metadata_t child_value(mkey, INDEX_UNINITIALIZED, data[0], CELL_UNINITIALIZED);
	  mandelbrotMap.insert(ckey, child_value);
	}
	
	{
	  int ixx = ix<<1 | 0x1;
	  int iyy = iy<<1;
	  amr_key_t ckey(compute_morton_key(ixx,iyy), encode_level_tree(level+1,treeId));

	  int mx = ixx << (level_max-level-1);
	  int my = iyy << (level_max-level-1);
	  morton_key_t mkey(compute_morton_key(mx,my),encode_tree(treeId));
	  
	  metadata_t child_value(mkey, INDEX_UNINITIALIZED, data[1], CELL_UNINITIALIZED);
	  mandelbrotMap.insert(ckey, child_value);
	}
	
	{
	  int ixx = ix<<1;
	  int iyy = iy<<1 | 0x1;
	  amr_key_t ckey(compute_morton_key(ixx,iyy), encode_level_tree(level+1,treeId));
	  
	  int mx = ixx << (level_max-level-1);
	  int my = iyy << (level_max-level-1);
	  morton_key_t mkey(compute_morton_key(mx,my),encode_tree(treeId));
	  
	  metadata_t child_value(mkey, INDEX_UNINITIALIZED, data[2], CELL_UNINITIALIZED);
	  mandelbrotMap.insert(ckey, child_value);
	}
	
	{
	  int ixx = ix<<1 | 0x1;
	  int iyy = iy<<1 | 0x1;
	  amr_key_t ckey(compute_morton_key(ixx,iyy), encode_level_tree(level+1,treeId));

	  int mx = ixx << (level_max-level-1);
	  int my = iyy << (level_max-level-1);
	  morton_key_t mkey(compute_morton_key(mx,my),encode_tree(treeId));
	  
	  metadata_t child_value(mkey, INDEX_UNINITIALIZED, data[3], CELL_UNINITIALIZED);
	  mandelbrotMap.insert(ckey, child_value);
	}
	
	
	// mark coarse cell for removal
	value.status = CELL_TO_BE_REMOVED;
	mandelbrotMap.value_at(i) = value;
	
      } // end if (r2<R2)

    } // end status != CELL_UNINITIALIZED
    
  } // insert_refined_cells

  KOKKOS_INLINE_FUNCTION
  void delete_old_cells(const int& i) const
  {

    // read key/value at current i
    amr_key_t key = mandelbrotMap.key_at(i);
    metadata_t value = mandelbrotMap.value_at(i);

    CellStatus status = value.status; 
    
    // if value is CELL_TO_BE_REMOVED, then call erase
    // remove cell at coarse level (erase method returns true / false depending
    // if current kernel is in between calls to begin_erase / end_erase
    if (status == CELL_TO_BE_REMOVED) {
      mandelbrotMap.erase(key); 
    }
    
  } // delete_old_cells

  KOKKOS_INLINE_FUNCTION
  void reset_status(const int& i) const
  {

    // read key/value at current i
    metadata_t value = mandelbrotMap.value_at(i);

    CellStatus status = value.status; 
    
    // if value is CELL_TO_BE_REMOVED, then call erase
    // remove cell at coarse level (erase method returns true / false depending
    // if current kernel is in between calls to begin_erase / end_erase
    if (status == CELL_UNINITIALIZED) {
      value.status = CELL_REGULAR;
      mandelbrotMap.value_at(i) = value;
    }
    
  } // reset_status
  
  /*
   * main mandelbrot functor entry point.
   */
  //! functor 
  KOKKOS_INLINE_FUNCTION
  void operator()(const int& i) const
  {

    // check if we have a valid (key,value) pair at current iterator i
    if (mandelbrotMap.valid_at(i)) {

      if (refine_mode == INSERT_REFINED_CELLS) { 
	insert_refined_cells(i);
      } else if (refine_mode == DELETE_OLD_CELLS) {
	delete_old_cells(i);
      } else if (refine_mode == RESET_STATUS) {
	reset_status(i);
      }
      
    } // end valid_at
    
  } // end operator()
  
}; // struct Refine

// =========================================================================
// =========================================================================
/**
 * dump mandelbrot set (given as an Kokkos::UnorderedMap).
 *
 * file format is vtk unstructured grid.
 */
void dump_vtk_2d(const std::string& filename,
		 MandelbrotMap mandelbrotMap)
{

  std::cout << "dump vtk 2d\n";
  std::cout << "mandelbrotMap.size()=" << mandelbrotMap.size() << "\n";
  
  int nbNodesPerCells = 4; // 2d

  // compute total number of cells
  uint64_t nbCells = mandelbrotMap.size();

  // compute total number of vertices
  uint64_t nbNodes = nbCells*nbNodesPerCells;
  
  // allocate a 2d device View of size : number of nodes by 3
  // (x,y,z) of a given node
  using Nodes_pos      = Kokkos::View<real_t*[3], Device>;
  using Nodes_pos_host = Nodes_pos::HostMirror;

  Nodes_pos nodes_pos = Nodes_pos("nodes_position",nbNodes);

  // convert nodes_pos to a std::vector (we will need to improve / avoid this later)
  std::vector<dyablo::Point<3> > nodes_coord(nbNodes);
  
  // Kokkos view of keys
  using KeyVec   = Kokkos::View<amr_key_t*, Device>;
  //using KeyVec_h = Kokkos::View<amr_key_t*, Device>::HostMirror;

  KeyVec keyVec = KeyVec("vector_of_keys",mandelbrotMap.size());

  using CellData = Kokkos::View<double*, Device>;

  CellData data_levels = CellData("data_levels",mandelbrotMap.size());
  CellData data_mkeys = CellData("data_mkeys",mandelbrotMap.size());
  CellData data_mindex = CellData("data_mindex",mandelbrotMap.size());
  CellData data_mandelbrot = CellData("data_mandelbrot",mandelbrotMap.size());
  
  // create a vector of keys + some example cell data
  Kokkos::parallel_scan
    ("copy_kokkos_unordered_map_to_view",
     mandelbrotMap.capacity(),
     KOKKOS_LAMBDA (const int& i,
		    int & ivec,
		    const bool final) {
      if (mandelbrotMap.valid_at(i))
	{
	  if (final) {
	    amr_key_t key = mandelbrotMap.key_at(i);
	    metadata_t value = mandelbrotMap.value_at(i);
	    keyVec(ivec) = key;
	    data_levels(ivec) = 1.0*key.get_level();
	    data_mkeys(ivec) = 1.0*value.key[0];
	    data_mindex(ivec) = 1.0*value.index;
	    data_mandelbrot(ivec) = value.data;
	  }
	  
	  ivec++;
	}
    });
  
  printf("keyVec size is %ld\n",keyVec.size());
  
  // get nodes position
  Kokkos::parallel_for
    ("get nodes positions",
     keyVec.extent(0), KOKKOS_LAMBDA (const int& i) {
      
      amr_key_t key = keyVec(i);
      uint64_t morton = key.get_morton();
      uint32_t ix = morton_extract_bits<2,IX>(morton);
      uint32_t iy = morton_extract_bits<2,IY>(morton);
      
      int level = key.get_level();
      int N = 1<<level;
      
      real_t dx = 1.0/N;
      real_t dy = 1.0/N;
      
      int64_t ii = i;
      nodes_pos(ii*nbNodesPerCells+0,IX) = ix*dx;
      nodes_pos(ii*nbNodesPerCells+0,IY) = iy*dy;
      nodes_pos(ii*nbNodesPerCells+0,IZ) = 0.0;
      
      nodes_pos(ii*nbNodesPerCells+1,IX) = ix*dx+dx;
      nodes_pos(ii*nbNodesPerCells+1,IY) = iy*dy;
      nodes_pos(ii*nbNodesPerCells+1,IZ) = 0.0;
      
      nodes_pos(ii*nbNodesPerCells+2,IX) = ix*dx+dx;
      nodes_pos(ii*nbNodesPerCells+2,IY) = iy*dy+dy;
      nodes_pos(ii*nbNodesPerCells+2,IZ) = 0.0;
      
      nodes_pos(ii*nbNodesPerCells+3,IX) = ix*dx;
      nodes_pos(ii*nbNodesPerCells+3,IY) = iy*dy+dy;
      nodes_pos(ii*nbNodesPerCells+3,IZ) = 0.0;
      
    });
  
  // convert nodes_pos to nodes_coord (On host with OpenMP)
  {
    Nodes_pos_host nodes_pos_host = Kokkos::create_mirror(nodes_pos);
    
    Kokkos::deep_copy(nodes_pos_host, nodes_pos);
    
    Kokkos::parallel_for
      (Kokkos::RangePolicy<Kokkos::OpenMP>(0,nbNodes),
       [&] (const int i) {
	nodes_coord[i] = {nodes_pos_host(i,IX),
			  nodes_pos_host(i,IY),
			  nodes_pos_host(i,IZ)};
      });
    
  }

  ConfigMap configMap("./test_io_vtk_2d.ini");
  configMap.setString("output", "outputPrefix", "mandelbrot_set");
  dyablo::io::VTKWriter vtkWriter(configMap, nbCells);
  vtkWriter.open_file();
  vtkWriter.write_header();
  vtkWriter.write_metadata(0,0.0);
  vtkWriter.write_piece_header(nbNodes);
  
  vtkWriter.write_geometry<2>(nodes_coord);
  vtkWriter.write_connectivity<2>();

  vtkWriter.open_data();
  // write cell data - meta + heavy
  //vtkWriter.write_cell_data("level", cell_levels);
  vtkWriter.write_cell_data("level",        data_levels);
  vtkWriter.write_cell_data("morton_index", data_mkeys);
  vtkWriter.write_cell_data("memory_index", data_mindex);
  vtkWriter.write_cell_data("mandelbrot",   data_mandelbrot);

  vtkWriter.close_data();
  
  // finaly closing the file !
  vtkWriter.write_piece_footer();
  vtkWriter.close_grid();
  vtkWriter.write_footer();
  vtkWriter.close_file();

} // dump_vtk_2d

// =========================================================================
// =========================================================================
/**
 * Driver function for Mandelbrot set computation.
 *
 */
void compute_mandelbrot_2d(int level_min,
			   int level_max,
			   int max_capacity_prefactor)
{
  
  std::cout << "=================================\n";
  std::cout << "===== compute Mandelbrot Set ====\n";
  std::cout << "=================================\n";

  std::cout << "Level min = " << level_min << "\n";
  std::cout << "Level max = " << level_max << "\n";
  
  // linear size along a direction
  int N = 1 << level_min;
  
  // an unordered map with metadata
  MandelbrotMap mandelbrotMap;

  // maximun capacity of the hash map container
  uint64_t total_capacity = max_capacity_prefactor*N*N; 
  
  std::cout << "Creating a metadata map with nLevels=" << level_min << " and capacity of " << total_capacity << " elements\n";
  
  // allocate some space in the hash map
  mandelbrotMap.rehash(total_capacity);

  // play with Kokkos API for UnorderedMap
  std::cout << "is mandelbrotMap insertable ? " << mandelbrotMap.is_insertable_map << "\n";
  std::cout << "is mandelbrotMap modifiable ? " << mandelbrotMap.is_modifiable_map << "\n";
  
  std::cout << "mandelbrotMap.size()     = " << mandelbrotMap.size() << std::endl;
  std::cout << "mandelbrotMap.capacity() = " << mandelbrotMap.capacity() << " (max size)" << std::endl;

  // just initialize mandelbrotMap with a coarse uniform grid
  FillCoarseMap fill(mandelbrotMap, level_min, level_max);

  std::cout << "[before refine] number of cells in mandelbrotMap "
	    << mandelbrotMap.size() << " " << "\n";
  
  // refine nbIter times
  Refine::apply(mandelbrotMap, level_min, level_max);
  Kokkos::fence();
  std::cout << "[after  refine] number of cells in mandelbrotMap "
	    << mandelbrotMap.size() << " " << "\n";

  uint64_t Nsquare = 1 << 2*level_max;
  std::cout << "Sparsity of Mandelbrot set : " << 100.0*mandelbrotMap.size() / Nsquare << "%\n";
  
  // create array (Kokkos::View) of keys
  using KeyVec   = Kokkos::View<amr_key_t*, Device>;

  // create array of morton key (TODO replace uint64_t by morton_key_t)
  using KeyVec2  = Kokkos::View<uint64_t*, Device>;
  //using KeyVec_h = Kokkos::View<amr_key_t*, Device>::HostMirror;

  KeyVec keyVec = KeyVec("keys",mandelbrotMap.size());
  KeyVec2 sorted_keyVec = KeyVec2("sorted_keys",mandelbrotMap.size());
  
  using Morton2AMR = Kokkos::UnorderedMap<uint64_t, amr_key_t, Device>;
  Morton2AMR morton2amr;
  morton2amr.rehash(mandelbrotMap.capacity());
  
  // fill keyVec, init sorted_keyVec and morton2amr
  Kokkos::parallel_scan
    ("copy_kokkos_unordered_map_to_view",
     mandelbrotMap.capacity(),
     KOKKOS_LAMBDA (const int& i,
		    int & ivec,
		    const bool final) {
      if(mandelbrotMap.valid_at(i))
	{
	  if(final) {
	    keyVec(ivec) = mandelbrotMap.key_at(i);
	    metadata_t value = mandelbrotMap.value_at(i);
	    
	    uint64_t mkey = value.key[0];
	    sorted_keyVec(ivec) = mkey;
	    morton2amr.insert(mkey, mandelbrotMap.key_at(i));
	  }
	  
	  ivec++;
	}
    });
  
  printf("keyVec size is %ld\n",keyVec.size());

  // for(size_t i=0; i<keyVec.size(); ++i)
  //   printf("%ld %ld \n",i,keyVec(i)[0]);

  // sort KeyVec
  //Kokkos::sort(keyVec);
  Kokkos::sort(sorted_keyVec);

  // not really sure there should be a fence here; without it data_levels
  // are not correct (to be analyzed, why exactly is it so)
  Kokkos::fence();

  // for(size_t i=0; i<keyVec.size(); ++i)
  //   printf("%ld %ld \n",i,keyVec2(i));

  // fill index in metadata hashmap
  Kokkos::parallel_for
    ("fill_indexed_in_metadata",
     sorted_keyVec.size(),
     KOKKOS_LAMBDA (const int& i) {
      
      // read morton key along Z-curve
      uint64_t mkey = sorted_keyVec(i);
      
      // find corresponding amr key by hash table loopkup
      size_t ikey = morton2amr.find(mkey);
      amr_key_t amr_key = morton2amr.value_at(ikey);
      
      // insert index into metadata
      ikey = mandelbrotMap.find(amr_key);
      
      // ikey should always be here
      metadata_t value = mandelbrotMap.value_at(ikey);
      
      value.index = i;
      
      mandelbrotMap.value_at(ikey) = value;
    });

  // dump in vtk file format
  dump_vtk_2d("mandelbrot_set_2d.vtk", mandelbrotMap);
  
} // compute_mandelbrot_2d

} // namespace dyablo

// =========================================================================
// =========================================================================
// =========================================================================
// =========================================================================
int main(int argc, char* argv[])
{

  // Create MPI session if MPI enabled
#ifdef DYABLO_USE_MPI
  dyablo::GlobalMpiSession mpiSession(&argc,&argv);
#endif // DYABLO_USE_MPI

  Kokkos::initialize(argc, argv);

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
  }


  // analyze command line arguments :
  // 
  // - level_min ==> coarse grid is (2**level_min)x(2**level_min)
  // - level_max
  // - max_capacity_prefactor : used to preallocated the hash table
  //   hash table size is coarse grid size x max_capacity_prefactor
  const int level_min = argc>1 ? std::atoi(argv[1]) : 6;
  const int level_max = argc>2 ? std::atoi(argv[2]) : 10;
  const int max_capacity_prefactor = argc>3 ?
    std::atoi(argv[3]) : 500;
  
  dyablo::compute_mandelbrot_2d(level_min,level_max,max_capacity_prefactor);

  Kokkos::finalize();

  return EXIT_SUCCESS;
  
} // end main
