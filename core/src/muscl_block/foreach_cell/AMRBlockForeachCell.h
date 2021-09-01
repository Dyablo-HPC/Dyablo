#pragma once

#include "muscl_block/foreach_cell/AMRBlockForeachCell_CellArray.h"

namespace dyablo {
namespace muscl_block {

/**
 * Constant data in AMRBlockForeachCell
 * This must be copiable to device
 **/
struct AMRBlockForeachCell_CData{
  uint32_t ndim;
  const LightOctree lmesh;
  uint32_t bx,by,bz;
  real_t dx_scale, dy_scale, dz_scale;
  uint32_t nbOctsPerGroup;
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
  AMRBlockForeachCell_CellMetaData(const AMRBlockForeachCell_CData& cdata)
  : cdata( cdata )
  {}

  KOKKOS_INLINE_FUNCTION
  pos_t getCellSize( const CellIndex& iCell ) const
  {
    const AMRBlockForeachCell_CData& cdata = this->cdata;
    const LightOctree& lmesh = cdata.lmesh;
    real_t oct_size = lmesh.getSize(iCell.iOct);

    return pos_t{
      oct_size * cdata.dx_scale,
      oct_size * cdata.dy_scale,
      oct_size * cdata.dz_scale
    };
  }
  
  KOKKOS_INLINE_FUNCTION
  pos_t getCellCenter( const CellIndex& iCell ) const
  {
    const AMRBlockForeachCell_CData& cdata = this->cdata;
    int ndim = cdata.ndim;
    const LightOctree& lmesh = cdata.lmesh;
    LightOctree::pos_t oct_center = lmesh.getCenter(iCell.iOct);
    pos_t cell_size = this->getCellSize(iCell);

    pos_t res{
      oct_center[IX] + ( (int32_t)(iCell.i) - (int32_t)(iCell.bx)/2 ) * cell_size[IX] + 0.5*cell_size[IX],
      oct_center[IY] + ( (int32_t)(iCell.j) - (int32_t)(iCell.by)/2 ) * cell_size[IY] + 0.5*cell_size[IY],
      oct_center[IZ] + ( (int32_t)(iCell.k) - (int32_t)(iCell.bz)/2 ) * cell_size[IZ] + 0.5*cell_size[IZ]
    };

    if(ndim == 2) res[IZ] = 0;

    return res;
  }

private:
  const AMRBlockForeachCell_CData cdata;
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
  using CellArray_global_ghosted = AMRBlockForeachCell_CellArray_impl::CellArray_global_ghosted;
  using CellMetaData = AMRBlockForeachCell_CellMetaData;
  friend CellMetaData;

  using Patch = typename PatchManager_t::Patch;
  using CellArray_patch = typename PatchManager_t::CellArray_patch;

private:
  template< typename View_t >
  using CellArray_base = AMRBlockForeachCell_CellArray_impl::CellArray_base<View_t>;
  using CData = AMRBlockForeachCell_CData;
  // Struct constant parameters for all patches  
  const CData cdata;
  PatchManager_t patchmanager;

public:
  // TODO : final list of parameters instead of cdata
  AMRBlockForeachCell_impl(const CData& cdata)
  : cdata(cdata), patchmanager(cdata)
  {}

  /**
   * Get CellMetaData related to current mesh
   **/
  CellMetaData getCellMetaData()
  {
    return AMRBlockForeachCell_CellMetaData(this->cdata);
  }

  /**
   * Create a new Patch::CellArray from a global array with ghost zone
   * The slice of U corresponding to the current patch will be exposed inside foreach_patch
   * Only CellIndexes inside the block are allowed for Arrays created with this method : 
   * when CellIndex is outside the block (when iter_space is ghosted, or an offset was applied), 
   * CellArray::convert_index returns an index with status == invalid (no neighbor search is performed)
   **/
  CellArray_global get_global_array(const DataArrayBlock& U, uint32_t gx, uint32_t gy, uint32_t gz, const id2index_t& fm)
  {
    const CData& cdata = this->cdata;
    uint32_t bx = cdata.bx+2*gx;
    uint32_t by = cdata.by+2*gy;
    uint32_t bz = cdata.bz+2*gz;
    assert(U.extent(0) == bx*by*bz);
    assert(U.extent(2) >= cdata.lmesh.getNumOctants());
    assert( cdata.ndim != 2 || bz==1 );

    return CellArray_global{U, bx, by, bz, (uint32_t)U.extent(2), fm};
  }
  /**
   * Create a new Patch::CellArray from a global array and its ghosts
   * The slice of U corresponding to the current patch will be exposed inside foreach_patch.
   * For CellIndexes outside the block (when iter_space is ghosted, or an offset was applied),
   * CellArray::convert_index may perform a neighbor search to find the corresponding ghost cell
   * (that might be non-conforming, see CellIndex documentation)
   **/
  CellArray_global_ghosted get_ghosted_array(const DataArrayBlock& U, const DataArrayBlock& Ughost, const LightOctree& lmesh, const id2index_t& fm)
  {
    const CData& cdata = this->cdata;
    uint32_t bx = cdata.bx;
    uint32_t by = cdata.by;
    uint32_t bz = cdata.bz;
    assert(U.extent(0) == bx*by*bz);
    assert(Ughost.extent(0) == bx*by*bz || Ughost.extent(2) == 0 );
    assert(U.extent(2) == cdata.lmesh.getNumOctants());
    assert( cdata.ndim != 2 || bz==1 );

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
  void foreach_patch(const std::string& kernel_name, const Function& f) const
  {
    patchmanager.foreach_patch(kernel_name, f);
  }

  /**
   * Same as a single foreach_cell inside foreach_patch with no temporaries
   * Patch policy is ignored here
   **/
  template <typename View_t, typename Function>
  void foreach_cell(const std::string& kernel_name, const CellArray_base<View_t>& iter_space, const Function& f) const
  {
    uint32_t bx = iter_space.bx;
    uint32_t by = iter_space.by;
    uint32_t bz = iter_space.bz;
    uint32_t nbCellsPerBlock = bx*by*bz;
    uint32_t nbOcts = cdata.lmesh.getNumOctants();

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
   * Call the user-defined function f for each cell and perform a reduction with the provided reducer
   * @param kernel_name name for the Kokkos kernel
   * @param iter_space the iCell parameter in f will take every valid position inside iter_space
   * @param f a const CellIndex& patch, Update_t::value_type& update -> void functor that is compatible with Kokkos
   *        This is usually a CELL_LAMBDA that performs and operation on a cell
   * @param reducer is a Kokkos reducer (eg: Kokkos::Sum<double>) this is the last parameter used in Kokkos::parallel_reduce
   *                (it cannot be just a scalar to perform the default sum operation)
   **/
  template <typename Function, typename View_t, typename Reducer_t>
  void reduce_cell(const std::string& kernel_name, const CellArray_base<View_t>& iter_space, const Function& f, const Reducer_t& reducer) const
  {
    uint32_t bx = iter_space.bx;
    uint32_t by = iter_space.by;
    uint32_t bz = iter_space.bz;
    uint32_t nbCellsPerBlock = bx*by*bz;
    uint32_t nbOcts = cdata.lmesh.getNumOctants();

    Kokkos::parallel_reduce( kernel_name, 
      Kokkos::RangePolicy<>(0,nbCellsPerBlock*nbOcts),
      KOKKOS_LAMBDA( uint32_t index, typename Reducer_t::value_type& update )
    {
      uint32_t iOct = index/nbCellsPerBlock;
      index = index%nbCellsPerBlock;

      uint32_t k = index/(bx*by);
      uint32_t j = (index - k*bx*by)/bx;
      uint32_t i = index - j*bx - k*bx*by;

      CellIndex iCell = {{iOct,false}, i, j, k, bx, by, bz};
      f( iCell, update );
    }, reducer);
  }
};

} // namespace muscl_block
} // namespace dyablo