#include "muscl_block/MapUserData.h"

#include "shared/kokkos_shared.h"
#include "shared/morton_utils.h"

namespace dyablo
{
namespace muscl_block
{

namespace{

/**
 * A GPU-compatible abstract interface to contain correspondance 
 * between octants before and after coarsening/refinement
 * 
 * The difficulty is to take into account the specificities of AMR :
 * - When an octant is refined, the user data that it contains must be copied to multiple new octants
 * - When and octant is coarsened, user data has to be gathered from multiple source octants
 * 
 * This interface can be used to query a list of pairs (see OctMapping type) to copy old octants to new octants
 * 
 * This implementation fetches the "mapping" from a PABLO octree and translates it into a 
 * GPU-compatible Kokkos::view.
 **/
class AMR_Remapper{
public:
  AMR_Remapper(std::shared_ptr<AMRmesh> amr_mesh)
  {
    /**
     * In this constructor we translate the results of ParaTree::getMapping()
     * and store it into a Kokkos::view so it can be used on device
     * The current algorithm is sequential because the relative position of 
     * the old octant compared to the new is deduced by the order in which the octants appear (morton order)
     * 
     * TODO : Optimize this!
     * Ideas : 
     *  - compute relative position without relying on octant order in list to be able to use OpenMP
     *  - modify PABLO to get direct access to the mapping list and deep_copy it directly
     *  - Use the hashmap in LightOctree_hashmap (before + after) to compute the mapping
     **/     

    constexpr int ndim = 3;
    uint32_t nbOcts = amr_mesh->getNumOctants();

    std::vector<OctMapping> data_vector;
    data_vector.reserve(1.1 * nbOcts);

    uint8_t child_id = 0; // Id of the child if we're refining
    for (uint32_t iOct=0; iOct < nbOcts; ++iOct)
    {
      std::vector<uint32_t> mapper;
      std::vector<bool> isghost;
      amr_mesh->getMapping(iOct, mapper, isghost);

      if ( amr_mesh->getIsNewC(iOct) ) 
      {
        for( size_t i=0; i<mapper.size(); i++ )
        {
          // We assume that octants are in morton order in the mapper vector
          uint8_t px = morton_extract_bits<ndim, IX>(i);
          uint8_t py = morton_extract_bits<ndim, IY>(i);
          uint8_t pz = morton_extract_bits<ndim, IZ>(i);

          data_vector.push_back({
            iOct, mapper[i],
            -1, {px, py, pz},
            isghost[i]
          });
        }
      }
      else if ( amr_mesh->getIsNewR(iOct) ) 
      {
        uint32_t iOct_src = mapper[0];
        assert( !isghost[0] );
        
        // iOct_src will appear 2^ndim times as a source value for all its children
        // We assume that his children iOct will apear in morton order
        uint8_t px = morton_extract_bits<ndim, IX>(child_id);
        uint8_t py = morton_extract_bits<ndim, IY>(child_id);
        uint8_t pz = morton_extract_bits<ndim, IZ>(child_id);

        data_vector.push_back({
            iOct, iOct_src,
            1, {px, py, pz},
            false
          });
        
        child_id++;
        if (child_id == std::pow(2, ndim) )
          child_id = 0;
      }
      else
      {
        uint32_t iOct_src = mapper[0];
        assert( !isghost[0] );
        data_vector.push_back({
            iOct, iOct_src,
            0
        });
      }
    }

    Kokkos::View<OctMapping*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > 
          data_host(data_vector.data(), data_vector.size());

    Kokkos::realloc(data, data_vector.size());
    Kokkos::deep_copy(data, data_host);
  }

  /// Get Number of mappings (octant pairs)
  KOKKOS_INLINE_FUNCTION
  uint32_t size() const
  {
    return data.size();
  }

  /**
   * Mapping from a source to a destination octant
   * Is a pair of octant with info about relative position between octants
   **/
  struct OctMapping{
    uint32_t iOct_dest, iOct_src; //! Source and destination octants
    int8_t level_diff; //! level difference between src and dest octants
                        //! (> 0 : src coarser and dest more refined)
    Kokkos::Array<uint8_t, 3> sub_pos; //! Position of smaller octant inside larger octant 
    bool isGhost; //! Source octant is ghost (Only for new coarse octant) 
  };

  /**
   *  Get a mapping in the list 
   * @returns see struct OctMapping
   **/
    KOKKOS_INLINE_FUNCTION
  OctMapping operator[](uint32_t i) const
  {
    return data(i);
  }

private:
  //TODO optimize that : should not use view of structs
  Kokkos::View<OctMapping*> data; 
};

/// Data to be accessed inside the Kokkos kernel
struct FunctorData{
  uint32_t nbFields;
  DataArrayBlock Usrc;
  DataArrayBlock Usrc_ghost;
  DataArrayBlock Udest;
  uint32_t bx, by, bz;
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
  int ndim = 3;
  uint32_t suboctant_count = std::pow( 2, level_diff*ndim ); 
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

} //namespace

void MapUserDataFunctor::apply( std::shared_ptr<AMRmesh> amr_mesh,
                                ConfigMap configMap,
                                blockSize_t blockSizes,
                                DataArrayBlock Usrc,
                                DataArrayBlock Usrc_ghost,
                                DataArrayBlock& Udest  )
{
  uint32_t nbOcts = amr_mesh->getNumOctants();
  uint32_t nbFields = Usrc.extent(1);
  uint32_t nbCellsPerOct = Usrc.extent(0);

  AMR_Remapper remap(amr_mesh);

  Udest = DataArrayBlock("U",  nbCellsPerOct, nbFields, nbOcts);

  const FunctorData d{
    nbFields,
    Usrc,
    Usrc_ghost,
    Udest,
    blockSizes[IX], blockSizes[IY], blockSizes[IZ]
  };

  // using kokkos team execution policy
  uint32_t nbTeams = configMap.getInteger("amr", "nbTeams", 16);
  using policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<uint32_t>>;
  Kokkos::parallel_for("SolverHydroMusclBlock::map_userdata_after_adapt",
                       policy_t(nbTeams, Kokkos::AUTO() ),
                       KOKKOS_LAMBDA( policy_t::member_type member )
  {
    for(uint32_t iPair = member.league_rank(); iPair < remap.size(); iPair+=member.league_size() )
    {
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
    }
  });

}

} // namespace muscl_block
} // namespace dyablo