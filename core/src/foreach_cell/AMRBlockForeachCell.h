#pragma once

#include "foreach_cell/AMRBlockForeachCell_CellArray.h"

namespace dyablo {


/**
 * Constant data in AMRBlockForeachCell
 * This must be copiable to device
 **/
struct AMRBlockForeachCell_CData{
  uint32_t ndim; /// Number of dimensions : 2D or 3D
  uint32_t bx,by,bz; /// Dimensions of cell blocks
  real_t xmin, ymin, zmin; /// min corner of physical domain
  real_t xmax, ymax, zmax; /// min corner of physical domain  
  uint32_t nbOctsPerGroup; /// Group size (Effect will vary between implementation, hint for temporary allocation)
};

/**
 * Information related to the position and size of cells
 * This can be captured on device and referenced inside foreach_patch/cell
 **/
class AMRBlockForeachCell_CellMetaData
{
public:
  using CellIndex = AMRBlockForeachCell_CellArray_impl::CellIndex;
  using pos_t = Kokkos::Array<real_t, 3>;

  inline
  AMRBlockForeachCell_CellMetaData(const AMRBlockForeachCell_CData& cdata, AMRmesh& pmesh)
  : cdata( cdata ), lmesh(pmesh.getLightOctree())
  {}

  /// Get the physical size of the cell
  KOKKOS_INLINE_FUNCTION
  pos_t getCellSize( const CellIndex& iCell ) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( iCell.is_valid(), "iCell should be valid to get size" );

    const AMRBlockForeachCell_CData& cdata = this->cdata;
    const LightOctree& lmesh = this->lmesh;
    auto oct_size = lmesh.getSize(iCell.iOct);

    real_t dx_scale = (cdata.xmax-cdata.xmin)/cdata.bx;
    real_t dy_scale = (cdata.ymax-cdata.ymin)/cdata.by;
    real_t dz_scale = (cdata.zmax-cdata.zmin)/cdata.bz;

    return pos_t{
      oct_size[IX] * dx_scale,
      oct_size[IY] * dy_scale,
      oct_size[IZ] * dz_scale
    };
  }
  
  /// Get the physical position of the center of the cell
  KOKKOS_INLINE_FUNCTION
  pos_t getCellCenter( const CellIndex& iCell ) const
  {
    DYABLO_ASSERT_KOKKOS_DEBUG( iCell.is_valid(), "iCell should be valid to get center" );

    const AMRBlockForeachCell_CData& cdata = this->cdata;
    int ndim = cdata.ndim;
    const LightOctree& lmesh = this->lmesh;
    LightOctree::pos_t oct_center = lmesh.getCenter(iCell.iOct);
    pos_t cell_size = this->getCellSize(iCell);

    real_t Lx = (cdata.xmax-cdata.xmin);
    real_t Ly = (cdata.ymax-cdata.ymin);
    real_t Lz = (cdata.zmax-cdata.zmin);

    pos_t res{
      cdata.xmin + oct_center[IX] * Lx + ( iCell.i - iCell.bx*0.5 + 0.5 ) * cell_size[IX],
      cdata.ymin + oct_center[IY] * Ly + ( iCell.j - iCell.by*0.5 + 0.5 ) * cell_size[IY],
      cdata.zmin + oct_center[IZ] * Lz + ( iCell.k - iCell.bz*0.5 + 0.5 ) * cell_size[IZ]
    };

    if(ndim == 2) res[IZ] = 0;

    return res;
  }

  KOKKOS_INLINE_FUNCTION
  CellIndex getCellFromPos( const pos_t& pos ) const
  {
    real_t xmin = cdata.xmin;
    real_t xmax = cdata.xmax;
    real_t ymin = cdata.ymin;
    real_t ymax = cdata.ymax;
    real_t zmin = cdata.zmin;
    real_t zmax = cdata.zmax;
    auto bx = cdata.bx; 
    auto by = cdata.by; 
    auto bz = cdata.bz; 

    // TODO refactor : copied from  ForeachParticle::distribute()
    real_t x = pos[IX];
    real_t y = pos[IY];
    real_t z = pos[IZ];
    real_t x01 = (x-xmin)/(xmax-xmin);
    real_t y01 = (y-ymin)/(ymax-ymin);
    real_t z01 = (z-zmin)/(zmax-zmin);
    assert( x01 > 0 && x01 < 1.0 );
    assert( y01 > 0 && y01 < 1.0 );
    assert( z01 >= 0 && z01 < 1.0 );

    LightOctree::OctantIndex iOct = lmesh.getiOctFromPos( {x01,y01,z01} );
    auto oct_size = lmesh.getSize( iOct );
    LightOctree::pos_t oct_corner = lmesh.getCorner( iOct );
    oct_corner[IX] = xmin + oct_corner[IX] * (xmax-xmin);
    oct_corner[IY] = ymin + oct_corner[IY] * (ymax-ymin);
    oct_corner[IZ] = zmin + oct_corner[IZ] * (zmax-zmin);      
    LightOctree::pos_t cell_size; 
    cell_size[IX] = oct_size[IX] * (xmax-xmin)/bx;
    cell_size[IY] = oct_size[IY] * (ymax-ymin)/by;
    cell_size[IZ] = oct_size[IZ] * (zmax-zmin)/bz;

    uint32_t ix = (x-oct_corner[IX])/cell_size[IX];
    uint32_t iy = (y-oct_corner[IY])/cell_size[IY];
    uint32_t iz = (z-oct_corner[IZ])/cell_size[IZ];      

    assert(!iOct.isGhost);

    return CellIndex{
      iOct, 
      ix, iy, iz, 
      bx, by, bz, 
      CellIndex::Status::LOCAL_TO_BLOCK 
    };  
  }
private:
  const AMRBlockForeachCell_CData cdata;
  LightOctree lmesh;
};

/**
 * Interface to iterate over AMR cells.
 * 
 * Introduces the Patch abstraction for hierarchical parallelism : 
 * patches are an arbitrary set of cells for which temporary arrays can be allocated
 * 
 * ```
 *    foreach patch
 *    | allocate temporary arrays
 *    | foreach cell in patch
 *    | | process cell
 * ```
 * 
 * Also provides abstract arrays containing cells
 * 
 * @tparam PatchManager_t provides the actual implementation of the Patch abstraction
 * AMRBlockForeachCell_impl class delegates temporary array management and iteration methods to PatchManager_t
 * 
 * NOTE : instances of this class cannot be copied to device (e.g. referenced inside foreach_patch/cell)
 **/
template< typename PatchManager_t >
class AMRBlockForeachCell_impl{
public:
  using CellIndex = AMRBlockForeachCell_CellArray_impl::CellIndex;
  using CellArray_global = AMRBlockForeachCell_CellArray_impl::CellArray_global;
  using CellArray_shape = AMRBlockForeachCell_CellArray_impl::CellArray_shape;
  using CellArray_global_ghosted = AMRBlockForeachCell_CellArray_impl::CellArray_global_ghosted;
  using CellMetaData = AMRBlockForeachCell_CellMetaData;
  friend CellMetaData;

  using Patch = typename PatchManager_t::Patch;
  using CellArray_patch = typename PatchManager_t::CellArray_patch;

private:
  template< typename View_t >
  using CellArray_base = AMRBlockForeachCell_CellArray_impl::CellArray_base<View_t>;
  using CData = AMRBlockForeachCell_CData;
  AMRmesh& pmesh;
  // Struct constant parameters for all patches 
  const CData cdata;
  PatchManager_t patchmanager;

public:
  AMRBlockForeachCell_impl( AMRmesh & pmesh, ConfigMap& configMap )
  :
    pmesh(pmesh),
    cdata{
      pmesh.getDim(),
      configMap.getValue<uint32_t>("amr", "bx", 0),
      configMap.getValue<uint32_t>("amr", "by", 0),
      configMap.getValue<uint32_t>("amr", "bz", 1),
      configMap.getValue<real_t>("mesh", "xmin", 0),
      configMap.getValue<real_t>("mesh", "ymin", 0),
      configMap.getValue<real_t>("mesh", "zmin", 0),
      configMap.getValue<real_t>("mesh", "xmax", 1),
      configMap.getValue<real_t>("mesh", "ymax", 1),
      configMap.getValue<real_t>("mesh", "zmax", 1),
      configMap.getValue<uint32_t>("amr", "nbOctsPerGroup", 1024)
    },
    patchmanager(cdata, pmesh)
  {}

  int getDim() const 
  {
    return cdata.ndim;
  }

  static constexpr bool has_blocks()
  {return true;}

  // Only available when has_blocks() == true
  Kokkos::Array<uint32_t, 3> blockSize() const
  {
    return {cdata.bx, cdata.by, cdata.bz };
  }

  uint32_t getNumCells() const 
  {
    return get_amr_mesh().getNumOctants() * cdata.bx * cdata.by * cdata.bz;
  }

  uint64_t getNumCells_global() const 
  {
    return get_amr_mesh().getGlobalNumOctants() * cdata.bx * cdata.by * cdata.bz;
  }

  AMRmesh& get_amr_mesh()
  {
    return pmesh;
  }

  const AMRmesh& get_amr_mesh() const
  {
    return pmesh;
  }

  /**
   * Get CellMetaData related to current mesh
   **/
  CellMetaData getCellMetaData() const
  {
    return AMRBlockForeachCell_CellMetaData(this->cdata, pmesh);
  }

  /**
   * Create a new Patch::CellArray from a global array with ghost zone
   * The slice of U corresponding to the current patch will be exposed inside foreach_patch
   * Only CellIndexes inside the block are allowed for Arrays created with this method : 
   * when CellIndex is outside the block (when iter_space is ghosted, or an offset was applied), 
   * CellArray::convert_index returns an index with status == invalid (no neighbor search is performed)
   **/
  // CellArray_global get_global_array(const DataArrayBlock& U, uint32_t gx, uint32_t gy, uint32_t gz, const id2index_t& fm)
  // {
  //   const CData& cdata = this->cdata;
  //   uint32_t bx = cdata.bx+2*gx;
  //   uint32_t by = cdata.by+2*gy;
  //   uint32_t bz = cdata.bz+2*gz;
  //   assert(U.extent(0) == bx*by*bz);
  //   assert(U.extent(2) >= pmesh.getNumOctants());
  //   assert( cdata.ndim != 2 || bz==1 );

  //   return CellArray_global{U, bx, by, bz, (uint32_t)U.extent(2), fm};
  // }

  CellArray_global_ghosted allocate_ghosted_array( std::string name, const FieldManager& fieldMgr)
  {
    const CData& cdata = this->cdata;
    uint32_t bx = cdata.bx;
    uint32_t by = cdata.by;
    uint32_t bz = cdata.bz;
    int nbCellsPerOct = bx*by*bz;
    int nbFields = fieldMgr.nbfields();
    int nbOcts = pmesh.getNumOctants();
    int nbGhosts = pmesh.getNumGhosts();
    auto fm = fieldMgr.get_id2index();

    DYABLO_ASSERT_HOST_RELEASE( cdata.ndim != 2 || bz==1, "bz should be 1 in 2D" );

    DataArrayBlock U(name, nbCellsPerOct, nbFields, nbOcts );
    DataArrayBlock Ughost(name+"ghost", nbCellsPerOct, nbFields, nbGhosts );

    const LightOctree& lmesh = pmesh.getLightOctree();

    return CellArray_global_ghosted(CellArray_global{U, bx, by, bz, (uint32_t)U.extent(2), fm}, Ughost, lmesh);
  }
  /**
   * Reserve a new temporary ghosted cell array local to each patch. 
   * Actual array has to be created with Patch::allocate_tmp() inside the foreach_patch loop.
   * (Actual allocation may happen anywhere between calls to reserve_patch_tmp() and allocate_tmp() )
   * Only CellIndexes inside the block (+ ghosts) are allowed for Arrays created with this method : 
   * For CellIndexes outside the block (when iter_space is has more ghosts, or an offset was applied),
   * CellArray::convert_index returns an index with status == invalid (no neighbor search is performed)
   **/
  typename CellArray_patch::Ref reserve_patch_tmp(std::string name, int gx, int gy, int gz, const id2index_t& fm, int nvars)
  {
    return patchmanager.reserve_patch_tmp(name, gx, gy, gz, fm, nvars);
  }

  /**
   * Call the user-defined function f for each patch
   * @param kernel_name name for the Kokkos kernel
   * @param f a const Patch& patch -> void functor that is compatible with Kokkos
   *        This is usually a KOKKOS_LAMBDA that calls patch.foreach_cells(...)
   **/
  template <typename Function>
  void foreach_patch(const std::string& kernel_name, const Function& f)
  {
    patchmanager.foreach_patch(kernel_name, f);
  }

  /**
   * Same as a single foreach_cell inside foreach_patch with no temporaries
   * Patch policy is ignored here
   **/
  template <typename Function>
  void foreach_cell(const std::string& kernel_name, const CellArray_shape& iter_space, const Function& f) const
  {
    uint32_t bx = iter_space.bx;
    uint32_t by = iter_space.by;
    uint32_t bz = iter_space.bz;
    uint32_t nbCellsPerBlock = bx*by*bz;
    uint32_t nbOcts = pmesh.getNumOctants();

    Kokkos::parallel_for( kernel_name, 
      Kokkos::RangePolicy<>(0,nbCellsPerBlock*nbOcts), 
      KOKKOS_LAMBDA( uint32_t index )
    {
      uint32_t iOct = index/nbCellsPerBlock;
      index = index%nbCellsPerBlock;

      uint32_t k = index/(bx*by);
      uint32_t j = (index - k*bx*by)/bx;
      uint32_t i = index - j*bx - k*bx*by;

      CellIndex iCell = {{iOct,false}, i, j, k, bx, by, bz};
      f( iCell );
    });
  }

  /**
   * Calls the user defined function f for each MPI-ghost in the current domain
   **/
  template <typename Function>
  void foreach_ghost_cell(const std::string& kernel_name, const CellArray_shape& iter_space, const Function& f) const
  {
    uint32_t bx = iter_space.bx;
    uint32_t by = iter_space.by;
    uint32_t bz = iter_space.bz;
    uint32_t nbCellsPerBlock = bx*by*bz;
    uint32_t nbGhosts = pmesh.getNumGhosts();

    Kokkos::parallel_for( kernel_name, 
      Kokkos::RangePolicy<>(0,nbCellsPerBlock*nbGhosts), 
      KOKKOS_LAMBDA( uint32_t index )
    {
      uint32_t iOct_ghost = index/nbCellsPerBlock;
      index = index%nbCellsPerBlock;

      uint32_t k = index/(bx*by);
      uint32_t j = (index - k*bx*by)/bx;
      uint32_t i = index - j*bx - k*bx*by;

      CellIndex iCell = {{iOct_ghost,true}, i, j, k, bx, by, bz};
      f( iCell );
    });
  }


  /**
   * Call the user-defined function f for each cell and perform a reduction with the provided reducer
   * @param kernel_name name for the Kokkos kernel
   * @param iter_space the iCell parameter in f will take every valid position inside iter_space
   * @param f a const CellIndex& patch, Update_t::value_type& update -> void functor that is compatible with Kokkos
   *        This is usually a CELL_LAMBDA that performs and operation on a cell
   * @param reducer is a Kokkos reducer (eg: Kokkos::Sum<double>) this is the last parameter used in Kokkos::parallel_reduce
   *                (it cannot be just a scalar to perform the default sum operation)
   **/
  template <typename Function, typename... Reducer_t>
  void reduce_cell(const std::string& kernel_name, const CellArray_shape& iter_space, const Function& f, const Reducer_t&... reducer) const
  {
    uint32_t bx = iter_space.bx;
    uint32_t by = iter_space.by;
    uint32_t bz = iter_space.bz;
    uint32_t nbCellsPerBlock = bx*by*bz;
    uint32_t nbOcts = pmesh.getNumOctants();

    Kokkos::parallel_reduce( kernel_name, 
      Kokkos::RangePolicy<>(0,nbCellsPerBlock*nbOcts),
      KOKKOS_LAMBDA( uint32_t index, typename Reducer_t::value_type&... update )
    {
      uint32_t iOct = index/nbCellsPerBlock;
      index = index%nbCellsPerBlock;

      uint32_t k = index/(bx*by);
      uint32_t j = (index - k*bx*by)/bx;
      uint32_t i = index - j*bx - k*bx*by;

      CellIndex iCell = {{iOct,false}, i, j, k, bx, by, bz};
      f( iCell, update... );
    }, reducer...);
  }

  /// Use Kokkos::Sum as default reducer for reduce_cell
  template <typename Function, typename... Value_t>
  void reduce_cell(const std::string& kernel_name, const CellArray_shape& iter_space, const Function& f, Value_t&... reducer) const
  {
    reduce_cell(kernel_name, iter_space, f, Kokkos::Sum<Value_t>(reducer)...);
  }
};


} // namespace dyablo