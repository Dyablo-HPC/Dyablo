#pragma once

#include "muscl_block/foreach_cell/AMRBlockForeachCell_group_CellArray.h"

#define PATCH_LAMBDA [&]
#define CELL_LAMBDA KOKKOS_LAMBDA

namespace dyablo {
namespace muscl_block {

namespace AMRBlockForeachCell_group_impl{

class AMRBlockForeachCell_Patch;
class AMRBlockForeachCell_CellMetaData;

class AMRBlockForeachCell{
public:
  struct CData{
    uint32_t ndim;
    const LightOctree lmesh;
    uint32_t bx,by,bz;
    real_t dx_scale, dy_scale, dz_scale;
    uint32_t nbOctsPerGroup;
  }; 
  // TODO : final list of parameters instead of cdata
  AMRBlockForeachCell(const CData& cdata);
private:
  // Struct constant parameters for all patches  
  const CData cdata;

public:

  using CellIndex = AMRBlockForeachCell_group_impl::CellIndex;
  using CellArray_global = AMRBlockForeachCell_group_impl::CellArray_global;
  using CellArray_global_ghosted = AMRBlockForeachCell_group_impl::CellArray_global_ghosted;
  using CellArray_patch = AMRBlockForeachCell_group_impl::CellArray_patch;


  using Patch = AMRBlockForeachCell_Patch;
  using CellMetaData = AMRBlockForeachCell_CellMetaData;
  friend CellMetaData;

  /**
   * Get CellMetaData related to current mesh
   * 
   * NOTE : This method should not be called on device (inside foreach_patch/cell)
   * Get a copy of the metadata before the loop and capture it with lambda. 
   **/
  CellMetaData getCellMetaData();

  /**
   * Create a new Patch::CellArray from a global array with ghost zone
   * The slice of U corresponding to the current patch will be exposed inside foreach_patch
   * Only CellIndexes inside the block are allowed for Arrays created with this method : 
   * when CellIndex is outside the block (when iter_space is ghosted, or an offset was applied), 
   * CellArray::convert_index returns an index with status == invalid (no neighbor search is performed)
   **/
  CellArray_global get_global_array(const DataArrayBlock& U, uint32_t gx, uint32_t gy, uint32_t gz, const id2index_t& fm);
  /**
   * Create a new Patch::CellArray from a global array and its ghosts
   * The slice of U corresponding to the current patch will be exposed inside foreach_patch.
   * For CellIndexes outside the block (when iter_space is ghosted, or an offset was applied),
   * CellArray::convert_index may perform a neighbor search to find the corresponding ghost cell
   * (that might be non-conforming, see CellIndex documentation)
   **/
  CellArray_global_ghosted get_ghosted_array(const DataArrayBlock& U, const DataArrayBlock& Ughost, const LightOctree& lmesh, const id2index_t& fm);
  /**
   * Reserve a new temporary ghosted cell array local to each patch. 
   * Actual array has to be created with Patch::allocate_tmp() inside the foreach_patch loop.
   * (Actual allocation may happen anywhere between calls to reserve_patch_tmp() and allocate_tmp() )
   * Only CellIndexes inside the block (+ ghosts) are allowed for Arrays created with this method : 
   * For CellIndexes outside the block (when iter_space is has more ghosts, or an offset was applied),
   * CellArray::convert_index returns an index with status == invalid (no neighbor search is performed) 
   **/
  CellArray_patch::Ref reserve_patch_tmp(std::string name, int gx, int gy, int gz, const id2index_t& fm, int nvars);

  /**
   * Call the user-defined function f for each patch
   * @param kernel_name name for the Kokkos kernel
   * @param f a const Patch& patch -> void functor that is compatible with Kokkos
   *        This is usually a KOKKOS_LAMBDA that calls patch.foreach_cells(...)
   **/
  template <typename Function>
  void foreach_patch(const std::string& kernel_name, const Function& f) const;
};

/**
 * Represents a group of cells to be iterated upon
 * This is an abstract interface for hyerarchical parallelism on cell arrays 
 * It enables the allocation of temporary arrays to store intermediate results 
 * for the current patch
 **/
class AMRBlockForeachCell_Patch{
public:
  friend AMRBlockForeachCell;
  friend AMRBlockForeachCell::CellArray_patch;

  using CellIndex = AMRBlockForeachCell::CellIndex;
  using CellArray_patch = AMRBlockForeachCell::CellArray_patch;
  using CData = AMRBlockForeachCell::CData;

  struct PData{
    const CData cdata;
    uint32_t group_begin, nbOctsInGroup;
    //Kokkos::TeamPolicy<Kokkos::IndexType<uint32_t>>::member_type member;
  };

  KOKKOS_INLINE_FUNCTION
  AMRBlockForeachCell_Patch( const PData& pdata );

  using pos_t = Kokkos::Array<real_t, 3>;

  /**
   * Apply the user-defined function f to every cell of the patch
   * @param iter_space : the iCell parameter in f will take every valid position inside iter_space
   * @param f : a const CellIndex& iCell -> void function 
   *            This is usually a lambda that reads and modify CellArrays at position iCell
   **/
  template <typename View_t, typename Function>
  KOKKOS_INLINE_FUNCTION
  void foreach_cell(const CellArray_base<View_t>& iter_space, const Function& f) const;

  KOKKOS_INLINE_FUNCTION
  CellArray_patch allocate_tmp( const CellArray_patch::Ref& array_ref ) const;
private : 
  PData pdata;
};

/**
 * Information related to the position and size of cells
 * This can be captured by KOKKOS_LAMBDAS ( unlike AMRBlockForeachCell )
 **/
class AMRBlockForeachCell_CellMetaData
{
public:
  using pos_t = Kokkos::Array<real_t, 3>;

  inline
  AMRBlockForeachCell_CellMetaData(const AMRBlockForeachCell& foreach_cell)
  : cdata( foreach_cell.cdata )
  {}

  KOKKOS_INLINE_FUNCTION
  pos_t getCellSize( const CellIndex& iCell ) const;
  
  KOKKOS_INLINE_FUNCTION
  pos_t getCellCenter( const CellIndex& iCell ) const;

private:
  const AMRBlockForeachCell::CData cdata;
};

inline
AMRBlockForeachCell::AMRBlockForeachCell(const CData& cdata)
  : cdata(cdata)
{}

inline
AMRBlockForeachCell_Patch::AMRBlockForeachCell_Patch(const AMRBlockForeachCell_Patch::PData& pdata)
  : pdata(pdata)
{}

inline
AMRBlockForeachCell_CellMetaData AMRBlockForeachCell::getCellMetaData()
{
  return AMRBlockForeachCell_CellMetaData(*this);
}

KOKKOS_INLINE_FUNCTION
AMRBlockForeachCell::CellMetaData::pos_t AMRBlockForeachCell::CellMetaData::getCellSize( const CellIndex& iCell ) const
{
  const AMRBlockForeachCell::CData& cdata = this->cdata;
  const LightOctree& lmesh = cdata.lmesh;
  real_t oct_size = lmesh.getSize(iCell.iOct);

  return pos_t{
    oct_size * cdata.dx_scale,
    oct_size * cdata.dy_scale,
    oct_size * cdata.dz_scale
  };
}

KOKKOS_INLINE_FUNCTION
AMRBlockForeachCell::CellMetaData::pos_t AMRBlockForeachCell::CellMetaData::getCellCenter( const CellIndex& iCell ) const
{
  const AMRBlockForeachCell::CData& cdata = this->cdata;
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

template <typename Function>
void AMRBlockForeachCell::foreach_patch(const std::string& kernel_name, const Function& f) const
{
  const CData& cdata = this->cdata;

  uint32_t nbOcts = cdata.lmesh.getNumOctants();
  uint32_t nbOctsPerGroup = cdata.nbOctsPerGroup;
  for(uint32_t iGroup = 0; iGroup <= nbOcts/nbOctsPerGroup; iGroup++ )
  {
    uint32_t group_begin = nbOctsPerGroup*iGroup;
    uint32_t nbOctsInGroup = std::min( nbOctsPerGroup, nbOcts-group_begin );

    f( Patch({cdata, group_begin, nbOctsInGroup}) );
  }
}

// template <typename Function>
// struct foreach_cell_functor{
//   uint32_t nbCellsPerBlock, bx, by, bz;
//   uint32_t group_begin;
//   Function f;

//   KOKKOS_INLINE_FUNCTION
//   void operator()(uint32_t index) const
//   {
//     uint32_t iOct = group_begin + index/nbCellsPerBlock;
//     index = index%nbCellsPerBlock;

//     AMRBlockForeachCell::CellIndex iCell = {{iOct,false}, index, bx, by, bz};
//     f( iCell );
//   }
// };

template <typename View_t, typename Function>
void AMRBlockForeachCell_Patch::foreach_cell(const CellArray_base<View_t>& iter_space, const Function& f) const
{
  uint32_t bx = iter_space.bx;
  uint32_t by = iter_space.by;
  uint32_t bz = iter_space.bz;

  uint32_t nbOctsInGroup = this->pdata.nbOctsInGroup;
  uint32_t group_begin = this->pdata.group_begin;

  // Kokkos::parallel_for( "AMRBlockForeachCell::Patch::foreach_cell",
  //     Kokkos::RangePolicy<>(0,nbOctsInGroup*nbCellsPerBlock), 
  //    foreach_cell_functor<Function>{nbCellsPerBlock, bx, by, bz, group_begin, f});

  // uint32_t vector_size = std::min( (uint32_t)Kokkos::TeamPolicy<>::vector_length_max(), bx );
  // Kokkos::parallel_for( "AMRBlockForeachCell::Patch::foreach_cell",
  //   Kokkos::TeamPolicy<>(nbOctsInGroup, Kokkos::AUTO(), vector_size ),
  //   KOKKOS_LAMBDA( auto m)
  // {
  //   uint32_t iOct = group_begin+m.league_rank();
  //   Kokkos::parallel_for( Kokkos::TeamThreadRange(m, bz*by),
  //     [&]( uint32_t kj )
  //   {
  //     uint32_t k = kj/by;
  //     uint32_t j = kj%by;
  //     Kokkos::parallel_for( Kokkos::ThreadVectorRange(m, bx),
  //     [&]( uint32_t i )
  //     {
  //       AMRBlockForeachCell::CellIndex iCell = {{iOct,false}, i, j, k, bx, by, bz};
  //       f( iCell );
  //     });
  //   });
  // });

  uint32_t nbCellsPerBlock = bx*by*bz;
  Kokkos::parallel_for( "AMRBlockForeachCell::Patch::foreach_cell",
      nbOctsInGroup*nbCellsPerBlock, 
      KOKKOS_LAMBDA (uint32_t index)
  {
    uint32_t iOct = group_begin + index/nbCellsPerBlock;
    index = index%nbCellsPerBlock;

    uint32_t k = index/(bx*by);
    uint32_t j = (index - k*bx*by)/bx;
    uint32_t i = index - j*bx - k*bx*by;

    AMRBlockForeachCell::CellIndex iCell = {{iOct,false}, i, j, k, bx, by, bz};
    f( iCell );
  });

  //_pdata.member.team_barrier();
}

inline
AMRBlockForeachCell::CellArray_global 
AMRBlockForeachCell::get_global_array(const DataArrayBlock& U, uint32_t gx, uint32_t gy, uint32_t gz, const id2index_t& fm)
{
  const CData& cdata = this->cdata;
  uint32_t bx = cdata.bx+2*gx;
  uint32_t by = cdata.by+2*gx;
  uint32_t bz = cdata.bz+2*gx;
  assert(U.extent(0) == bx*by*bz);
  assert(U.extent(2) >= cdata.lmesh.getNumOctants());
  assert( cdata.ndim != 2 || bz==1 );

  return CellArray_global{U, bx, by, bz, (uint32_t)U.extent(2), fm};
}

inline
AMRBlockForeachCell::CellArray_global_ghosted 
AMRBlockForeachCell::get_ghosted_array(const DataArrayBlock& U, const DataArrayBlock& Ughost,  const LightOctree& lmesh, const id2index_t& fm)
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

inline
AMRBlockForeachCell::CellArray_patch::Ref
AMRBlockForeachCell::reserve_patch_tmp(std::string name, int gx, int gy, int gz, const id2index_t& fm, int nvars)
{
  const CData& cdata = this->cdata;
  uint32_t bx = cdata.bx+2*gx;
  uint32_t by = cdata.by+2*gy;
  uint32_t bz = cdata.bz+2*gz;
  uint32_t nbOctsPerGroup = cdata.nbOctsPerGroup;
  assert( cdata.ndim != 2 || bz==1 );

  // Do not initialize View to improve first-touch behavior
  DataArrayBlock data(Kokkos::ViewAllocateWithoutInitializing(name), bx*by*bz, nvars, nbOctsPerGroup);
  return CellArray_patch({ data, bx, by, bz, (uint32_t)data.extent(2), fm });
}

AMRBlockForeachCell::CellArray_patch AMRBlockForeachCell::Patch::allocate_tmp(const AMRBlockForeachCell::CellArray_patch::Ref& array_ref) const
{
  return array_ref;
}

} // namespace AMRBlockForeachCell_group_impl

using AMRBlockForeachCell_group = AMRBlockForeachCell_group_impl::AMRBlockForeachCell;

} // namespace muscl_block
} // namespace dyablo