#pragma once

#include "foreach_cell/AMRBlockForeachCell.h"

#define PATCH_LAMBDA KOKKOS_LAMBDA
#define CELL_LAMBDA [&]
#define SCRATCH_LEVEL 1

namespace dyablo {


namespace AMRBlockForeachCell_scratch_impl{

using namespace AMRBlockForeachCell_CellArray_impl;
using CData = AMRBlockForeachCell_CData;
using policy_t = Kokkos::TeamPolicy<>;

class PatchManager;

class CellArray_patch : public CellArray_global
{
public:
  using Ref = CellArray_patch;

  CellArray_patch(){};
  CellArray_patch(const CellArray_global& a, uint32_t nbVars) : CellArray_global(a), nbVars(nbVars) {}

  uint32_t nbVars;
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
    uint32_t iOct;
    policy_t::member_type team_member;
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
  KOKKOS_INLINE_FUNCTION
  void foreach_cell(const CellArray_base<View_t>& iter_space, const Function& f) const
  {
    uint32_t bx = iter_space.bx;
    uint32_t by = iter_space.by;
    uint32_t bz = iter_space.bz;

    uint32_t iOct = pdata.iOct;
    Kokkos::parallel_for( Kokkos::TeamThreadRange(pdata.team_member, bz*by),
      [&]( uint32_t kj )
    {
      uint32_t k = kj/by;
      uint32_t j = kj%by;
      Kokkos::parallel_for( Kokkos::ThreadVectorRange(pdata.team_member, bx),
      [&]( uint32_t i )
      {
        CellIndex iCell = {{iOct,false}, i, j, k, bx, by, bz};
        f( iCell );
      });
    });

    pdata.team_member.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  CellArray_patch allocate_tmp( const CellArray_patch::Ref& array_ref ) const
  {
    CellArray_patch res = array_ref;
    res.U = CellArray_patch::View_t(this->pdata.team_member.team_scratch(SCRATCH_LEVEL), array_ref.bx*array_ref.by*array_ref.bz, array_ref.nbVars, 1);
    return res;
  }
};

class PatchManager{
public:
  using Patch = AMRBlockForeachCell_scratch_impl::Patch;
  using CellArray_patch = AMRBlockForeachCell_scratch_impl::CellArray_patch;

private:
  const CData cdata;
  const AMRmesh& pmesh;
  uint32_t scratch_size = 0;

public:
  PatchManager(const CData& cdata, const AMRmesh& pmesh)
  : cdata(cdata), pmesh(pmesh)
  {}

  CellArray_patch::Ref reserve_patch_tmp(std::string name, int gx, int gy, int gz, const id2index_t& fm, int nvars)
  {
    const CData& cdata = this->cdata;
    uint32_t bx = cdata.bx+2*gx;
    uint32_t by = cdata.by+2*gy;
    uint32_t bz = cdata.bz+2*gz;
    assert( cdata.ndim != 2 || bz==1 );

    scratch_size += DataArrayBlock::shmem_size(bx*by*bz, nvars, 1);
    return CellArray_patch({ DataArrayBlock(), bx, by, bz, 1, fm }, nvars);
  }  
  
  template <typename Function>
  void foreach_patch(const std::string& kernel_name, const Function& f)
  {
    std::cout << this->scratch_size << std::endl;
    const CData& cdata = this->cdata;
    uint32_t nbOcts = pmesh.getNumOctants();

    Kokkos::parallel_for( "AMRBlockForeachCell::Patch::foreach_patch",
      policy_t(nbOcts, Kokkos::AUTO())
        .set_scratch_size(SCRATCH_LEVEL, Kokkos::PerTeam(this->scratch_size)),
      KOKKOS_LAMBDA( policy_t::member_type team_member )
    {
      uint32_t iOct = team_member.league_rank();
      f( Patch({cdata, iOct, team_member}) ); 
    });

    scratch_size = 0;
  }
};

} // namespace AMRBlockForeachCell_scratch_impl

using AMRBlockForeachCell_scratch = AMRBlockForeachCell_impl<AMRBlockForeachCell_scratch_impl::PatchManager>;


} // namespace dyablo