#include "MapUserData.h"

#include "shared/amr/AMR_Remapper.h"

namespace dyablo
{
namespace muscl_block
{

namespace{

using AMR_Remapper = dyablo::shared::AMR_Remapper;

/// Data to be accessed inside the Kokkos kernel
struct FunctorData{
  uint32_t nbFields;
  DataArrayBlock Usrc;
  DataArrayBlock Usrc_ghost;
  DataArrayBlock Udest;
  uint32_t bx, by, bz;
  uint8_t ndim;
};

/**
 * Fill cell iCell in destination octant m.iOct_dest with 
 * data from octant m.iOct_src when both octants have same size
 **/
KOKKOS_INLINE_FUNCTION
void fill_cell_same_size( const FunctorData& d, 
                          const AMR_Remapper::OctMapping& m,
                          uint32_t iCell)
{
  for( uint32_t iField=0; iField<d.nbFields; iField++ )
  {
    d.Udest(iCell, iField, m.iOct_dest) = d.Usrc(iCell, iField, m.iOct_src);
  }
}

/**
 * Compute position of cell in coarse octant form cell position in more refined octant
 * @param iCell_small refined octant
 * @param bx, by, bz block size
 * @param level_diff amr level difference between coarse and refined cell 
 * @param sub_pos position of smaller octant inside bigger octant
 *                 _______ 
 *                |_|_|_|_|  Octant O is the smaller octant with 
 *                |_|_|_|_|  level_diff=2 and sub_pos = {2,1} 
 *                |_|_|O|_|    ^
 *                |_|_|_|_|  y |
 *                              -->
 *                               x
 *                -- End of description of param sub_pos --
 * @returns The index of the cell in the bigger octant that contains the cell iCell_small
 *          in smallest octant.
 **/
KOKKOS_INLINE_FUNCTION
uint32_t get_iCell_in_bigger_cell(uint32_t iCell_small, 
                                  uint32_t bx, uint32_t by, uint32_t bz, 
                                  uint8_t level_diff,
                                  const Kokkos::Array<uint8_t, 3>& sub_pos)
{
  coord_t pos_small = index_to_coord(iCell_small, bx, by, bz);
  uint32_t suboctant_count = std::pow( 2, level_diff ); 
  coord_t pos_big = {
    // Compute position of cell inside a grid at finer level in coarse block,
    // then divide according to level difference
    (pos_small[IX] + sub_pos[IX]*bx) / suboctant_count,
    (pos_small[IY] + sub_pos[IY]*by) / suboctant_count,
    (pos_small[IZ] + sub_pos[IZ]*bz) / suboctant_count
  };
  uint32_t iCell_big = coord_to_index(pos_big, bx, by, bz);

  return iCell_big;
}

/**
 * Fill cell iCell in destination octant iOct_dest with 
 * data from octant iOct_src when new octant is smaller
 * This works even if level difference is > than 1
 **/
KOKKOS_INLINE_FUNCTION
void fill_cell_newRefined(const FunctorData& d, 
                          const AMR_Remapper::OctMapping& m,
                          uint32_t iCell_dst)
{
  uint32_t iCell_src = get_iCell_in_bigger_cell(iCell_dst, 
                                                d.bx, d.by, d.bz,
                                                m.level_diff, m.sub_pos);

  for( uint32_t iField=0; iField<d.nbFields; iField++ )
  {
    d.Udest(iCell_dst, iField, m.iOct_dest) = d.Usrc(iCell_src, iField, m.iOct_src);
  }
}

/**
 * Fill cell iCell in destination octant iOct_dest with 
 * data from octant iOct_src when new octant is bigger
 * 
 * This works even if level difference is > than 1 
 * (This should not happen with PABLO)
 **/
KOKKOS_INLINE_FUNCTION
void fill_cell_newCoarse( const FunctorData& d, 
                          const AMR_Remapper::OctMapping& m,
                          uint32_t iCell_src)
{
  uint8_t level_diff = -m.level_diff;
  uint32_t iCell_dst = get_iCell_in_bigger_cell(iCell_src, 
                                                d.bx, d.by, d.bz,
                                                level_diff, m.sub_pos);
  uint32_t suboctant_count = std::pow( 2, level_diff*d.ndim ); 
  for( uint32_t iField=0; iField<d.nbFields; iField++ )
  {
    // Source can be a ghost when dst has just been coarsened
    // (This can't happen otherwise)
    real_t vsrc;
    if( m.isGhost )
      vsrc = d.Usrc_ghost(iCell_src, iField, m.iOct_src )/suboctant_count; 
    else
      vsrc = d.Usrc(iCell_src, iField, m.iOct_src )/suboctant_count;
    // Atomic is needed here because multiple cubcells accumulate their contribution 
    Kokkos::atomic_add(&d.Udest(iCell_dst, iField, m.iOct_dest), vsrc);
  }
}

void apply_aux( const AMR_Remapper& remap,
                uint8_t ndim,
                blockSize_t blockSizes,
                DataArrayBlock Usrc,
                DataArrayBlock Usrc_ghost,
                DataArrayBlock& Udest  )
{
  uint32_t nbOcts = remap.getNumOctants();
  uint32_t nbFields = Usrc.extent(1);
  uint32_t nbCellsPerOct = Usrc.extent(0);

  Kokkos::realloc(Udest, nbCellsPerOct, nbFields, nbOcts);

  const FunctorData d{
    nbFields,
    Usrc,
    Usrc_ghost,
    Udest,
    blockSizes[IX], blockSizes[IY], blockSizes[IZ],
    ndim
  };

  // using kokkos team execution policy
  using policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<uint32_t>>;
  Kokkos::parallel_for("SolverHydroMusclBlock::map_userdata_after_adapt",
                       policy_t(remap.size(), Kokkos::AUTO() ),
                       KOKKOS_LAMBDA( policy_t::member_type member )
  {
    uint32_t iPair = member.league_rank();
    AMR_Remapper::OctMapping mapping = remap[iPair];

    Kokkos::parallel_for( Kokkos::TeamVectorRange(member, nbCellsPerOct),
                          [&](uint32_t iCell)
    {
      if( mapping.level_diff == 0 )
      {
        fill_cell_same_size(d, mapping, iCell);
      }
      else if( mapping.level_diff < 0 )
      {
        fill_cell_newCoarse(d, mapping, iCell);
      }
      else if( mapping.level_diff > 0 )
      {
        fill_cell_newRefined(d, mapping, iCell);
      }
      else assert(false);
    });
  });

}

} //namespace

void MapUserDataFunctor::apply( const LightOctree_hashmap& lmesh_old,
                                const LightOctree_hashmap& lmesh_new,
                                blockSize_t blockSizes,
                                DataArrayBlock Usrc,
                                DataArrayBlock Usrc_ghost,
                                DataArrayBlock& Udest  )
{
  apply_aux( AMR_Remapper(lmesh_old, lmesh_new), lmesh_new.getNdim(),
             blockSizes, Usrc, Usrc_ghost, Udest );
}

void MapUserDataFunctor::apply( const LightOctree_pablo& lmesh_old,
                                const LightOctree_pablo& lmesh_new,
                                blockSize_t blockSizes,
                                DataArrayBlock Usrc,
                                DataArrayBlock Usrc_ghost,
                                DataArrayBlock& Udest  )
{
  apply_aux( AMR_Remapper(lmesh_new.getMesh()), lmesh_new.getNdim(),
             blockSizes, Usrc, Usrc_ghost, Udest );
}

} // namespace muscl_block
} // namespace dyablo