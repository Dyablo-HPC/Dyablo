#pragma once

#include "muscl_block/foreach_cell/AMRBlockForeachCell.h"

#define PATCH_LAMBDA [&]
#define CELL_LAMBDA KOKKOS_LAMBDA

namespace dyablo {
namespace muscl_block {

namespace AMRBlockForeachCell_group_impl{

using namespace AMRBlockForeachCell_CellArray_impl;
using CData = AMRBlockForeachCell_CData;

class PatchManager;

class CellArray_patch : public CellArray_global
{
public:
  using Ref = CellArray_patch;

  CellArray_patch(){};
  CellArray_patch(const CellArray_global& a) : CellArray_global(a) {}
};

/**
 * Represents a group of cells to be iterated upon
 * This is an abstract interface for hyerarchical parallelism on cell arrays 
 * It enables the allocation of temporary arrays to store intermediate results 
 * for the current patch
 **/
class Patch{
  friend PatchManager;
public:
  struct PData{
    const CData cdata;
    uint32_t group_begin, nbOctsInGroup;
    //Kokkos::TeamPolicy<Kokkos::IndexType<uint32_t>>::member_type member;
  };

private: 
  PData pdata;

public:
  KOKKOS_INLINE_FUNCTION
  Patch( const PData& pdata )
    : pdata(pdata)
  {}

  /**
   * Apply the user-defined function f to every cell of the patch
   * @param iter_space : the iCell parameter in f will take every valid position inside iter_space
   * @param f : a const CellIndex& iCell -> void function 
   *            This is usually a lambda that class CellArray_patch;reads and modify CellArrays at position iCell
   **/
  template <typename View_t, typename Function>
  void foreach_cell(const CellArray_base<View_t>& iter_space, const Function& f) const
  {
    uint32_t bx = iter_space.bx;
    uint32_t by = iter_space.by;
    uint32_t bz = iter_space.bz;

    uint32_t nbOctsInGroup = this->pdata.nbOctsInGroup;
    uint32_t group_begin = this->pdata.group_begin;

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

      CellIndex iCell = {{iOct,false}, i, j, k, bx, by, bz};
      f( iCell );
    });

    //_pdata.member.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  CellArray_patch allocate_tmp( const CellArray_patch::Ref& array_ref ) const
  {
    return array_ref;
  }
};

class PatchManager{
public:
  using Patch = AMRBlockForeachCell_group_impl::Patch;
  using CellArray_patch = AMRBlockForeachCell_group_impl::CellArray_patch;

private:
  const CData cdata;

public:
  PatchManager(const CData& cdata)
  : cdata(cdata)
  {}

  CellArray_patch::Ref reserve_patch_tmp(std::string name, int gx, int gy, int gz, const id2index_t& fm, int nvars)
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
  
  template <typename Function>
  void foreach_patch(const std::string& kernel_name, const Function& f) const
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

  // template <typename Function, typename View_t, typename Reducer_t>
  // void reduce_cell(const std::string& kernel_name, const CellArray_base<View_t>& iter_space, const Function& f, const Reducer_t& reducer) const
  // {
  //   this->foreach_patch(kernel_name, [&](const Patch& patch)
  //   {
  //     uint32_t bx = iter_space.bx;
  //     uint32_t by = iter_space.by;
  //     uint32_t bz = iter_space.bz;
  //     uint32_t nbCellsPerBlock = bx*by*bz;

  //     uint32_t nbOctsInGroup = patch.pdata.nbOctsInGroup;
  //     uint32_t group_begin = patch.pdata.group_begin;

  //     typename Reducer_t::value_type val;
  //     Reducer_t reducer_local(val);

  //     Kokkos::parallel_reduce( kernel_name,
  //         Kokkos::RangePolicy<>(0,nbOctsInGroup*nbCellsPerBlock), 
  //         KOKKOS_LAMBDA (uint32_t index, typename Reducer_t::value_type& update)
  //     {
  //       uint32_t iOct = group_begin + index/nbCellsPerBlock;
  //       index = index%nbCellsPerBlock;

  //       uint32_t k = index/(bx*by);
  //       uint32_t j = (index - k*bx*by)/bx;
  //       uint32_t i = index - j*bx - k*bx*by;

  //       CellIndex iCell = {{iOct,false}, i, j, k, bx, by, bz};
  //       f( iCell, update );
  //     }, reducer_local );
  //     reducer.join( reducer.reference(), val);
  //   });
  // }
};

} // namespace AMRBlockForeachCell_group_impl

using AMRBlockForeachCell_group = AMRBlockForeachCell_impl<AMRBlockForeachCell_group_impl::PatchManager>;

} // namespace muscl_block
} // namespace dyablo