#include "muscl_block/MapUserData.h"

#include "shared/kokkos_shared.h"
#include "shared/morton_utils.h"
#include "shared/LightOctree.h"

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
 **/
class AMR_Remapper{
public:
  /**
   * In this constructor we use the hashmap in LightOctree_hashmap to 
   * determine which octant(s) in the old amr mesh are related to the 
   * octants in the new AMR mesh.
   * 
   * This constructor runs on GPU with data already on device and doesn't need to query mappings from PABLO
   **/ 
  AMR_Remapper(const LightOctree_hashmap& lmesh_old, const LightOctree_hashmap& lmesh_new)
  {
    init(lmesh_old, lmesh_new);
  }
  
  void init(const LightOctree_hashmap& lmesh_old, const LightOctree_hashmap& lmesh_new)
  {
    //Init is a separate function because KOKKOS_LAMBDA cannot be user in the constructor

    // Optim idea : maybe use morton index instead of physical positions

    int ndim = lmesh_new.getNdim();
    int nsuboctants = (ndim==3) ? 8 : 4;
    uint32_t nbOcts = lmesh_new.getNumOctants();
    this->nbOcts = nbOcts;
    auto& data = this->data;

    // Scan the mappings to get total number of pairs and offset for each pair
    Kokkos::View<uint32_t*> offset("offset", nbOcts+1);
    Kokkos::parallel_scan(nbOcts,
                          KOKKOS_LAMBDA(const uint32_t iOct_new, int& update, const bool final)
    {
      // Hashmap allows us to do : old octant -> center position -> new octant
      LightOctree_hashmap::pos_t c_new = lmesh_new.getCenter({iOct_new, false});
      LightOctree_hashmap::OctantIndex iOct_old = lmesh_old.getiOctFromPos(c_new);
      uint8_t l_new = lmesh_new.getLevel({iOct_new, false});
      uint8_t l_old = lmesh_old.getLevel(iOct_old);

      // Multiple source octants when new coarsened cell
      int local_count = l_old > l_new ? nsuboctants : 1;
      
      update += local_count;
      if(final)
      {
        offset(iOct_new+1) = update;
      }
    });

    // Copy number of pairs (scalar) back to CPU to allocate data to the right size
    Kokkos::View<uint32_t> nbPairs_view = Kokkos::subview(offset,nbOcts);
    Kokkos::View<uint32_t>::HostMirror nbPairs_view_host = Kokkos::create_mirror_view(nbPairs_view);
    Kokkos::deep_copy(nbPairs_view_host, nbPairs_view);
    uint32_t nbPairs = nbPairs_view_host();
    Kokkos::realloc(data, nbPairs);

    // Fill `data` with the octants pairs
    Kokkos::parallel_for("AMR_Remapper::construct",
                         nbOcts,
                         KOKKOS_LAMBDA(uint32_t iOct_new)
    {
      LightOctree_hashmap::pos_t c_new = lmesh_new.getCenter({iOct_new, false});
      LightOctree_hashmap::OctantIndex iOct_old = lmesh_old.getiOctFromPos(c_new);
      uint8_t l_new = lmesh_new.getLevel({iOct_new, false});
      uint8_t l_old = lmesh_old.getLevel(iOct_old);

      if( l_new == l_old ) // Same size cell
      {
        assert(!iOct_old.isGhost); //Cannot be ghost
        data(offset(iOct_new)) = {
            iOct_new, iOct_old.iOct,
            0
          };
      }
      else if( l_new > l_old ) // New refined cell
      {
        assert(!iOct_old.isGhost); //Cannot be ghost

        LightOctree_hashmap::pos_t c_old = lmesh_old.getCenter(iOct_old);
        // Get relative position using physical position of both octants
        uint8_t px = static_cast<int>(c_old[IX] < c_new[IX]);
        uint8_t py = static_cast<int>(c_old[IY] < c_new[IY]);
        uint8_t pz = (ndim-2)*static_cast<int>(c_old[IZ] < c_new[IZ]);
        data(offset(iOct_new)) = {
            iOct_new, iOct_old.iOct,
            1, {px, py, pz},
            false
        };
      }
      else //if( l_new < l_old ) // New coarsened cell
      {
        real_t suboctant_size = lmesh_old.getSize(iOct_old);
        for( int i=0; i<nsuboctants; i++ )
        {
          uint8_t pz = i/4;
          uint8_t py = (i-pz*4)/2;
          uint8_t px = i - pz*4 - py*2;
          // Physical position of suboctant at relative position {px, py, pz}
          LightOctree_hashmap::pos_t c_suboctant = {
            c_new[IX] + px * suboctant_size - suboctant_size/2,
            c_new[IY] + py * suboctant_size - suboctant_size/2,
            (c_new[IZ] + pz * suboctant_size - suboctant_size/2) * (ndim-2)
          };

          LightOctree_hashmap::OctantIndex iOct_old = lmesh_old.getiOctFromPos(c_suboctant);

          data(offset(iOct_new)+i) = {
            iOct_new, iOct_old.iOct,
            -1, {px, py, pz},
            iOct_old.isGhost
          };
        }
      }
    });
  }

  AMR_Remapper(std::shared_ptr<AMRmesh> amr_mesh)
  {
    /**
     * In this constructor we translate the results of ParaTree::getMapping()
     * and store it into a Kokkos::view so it can be used on device
     * 
     * Optimization ideas : 
     *  - modify PABLO to get direct access to the mapping list and deep_copy it directly
     *  - Use the hashmap in LightOctree_hashmap (before + after) to compute the mapping
     **/     

    int ndim = amr_mesh->getDim();
    size_t nsuboctants = (ndim==3) ? 8 : 4;
    uint32_t nbOcts = amr_mesh->getNumOctants();
    this->nbOcts = nbOcts;

    // Scan the mappings to get total number of pairs and offset for each pair
    Kokkos::View<uint32_t*>::HostMirror offset("offset", nbOcts+1);
    Kokkos::parallel_scan(Kokkos::RangePolicy<Kokkos::OpenMP>( 0, nbOcts),
                          [=](const int iOct, int& update, const bool final)
    {
      int local_count = amr_mesh->getIsNewC(iOct) ? nsuboctants : 1;
      
      update += local_count;
      if(final)
      {
        offset(iOct+1) = update;
      }
    });

    uint32_t nbPairs = offset(nbOcts);
    Kokkos::realloc(data, nbPairs);
    auto data_host = Kokkos::create_mirror_view(data);

    Kokkos::parallel_for("AMR_Remapper::construct",
                         Kokkos::RangePolicy<Kokkos::OpenMP>( 0, nbOcts),
                         [=](uint32_t iOct)
    {
      std::vector<uint32_t> mapper;
      std::vector<bool> isghost;
      amr_mesh->getMapping(iOct, mapper, isghost);

      if ( amr_mesh->getIsNewC(iOct) ) // Coarsened octant
      {
        assert( mapper.size() == nsuboctants);

        for( size_t i=0; i<mapper.size(); i++ )
        {
          // We assume that octants are in morton order in the mapper vector
          uint8_t pz = i/4;
          uint8_t py = (i-pz*4)/2;
          uint8_t px = i - pz*4 - py*2;

          data_host(offset(iOct)+i) = {
            iOct, mapper[i],
            -1, {px, py, pz},
            isghost[i]
          };
        }
      }
      else if ( amr_mesh->getIsNewR(iOct) ) // Refined octant
      {
        assert( mapper.size() == 1);

        uint32_t iOct_src = mapper[0];
        assert( !isghost[0] );
        
        // iOct_src will appear 2^ndim times as a source value for all its children
        // Find relative position according to parity of the discrete position of the octant
        auto pos = amr_mesh->getCenter(iOct);
        real_t oct_size = amr_mesh->getSize(iOct);

        uint8_t px = static_cast<int>(pos[IX] / oct_size) % 2;
        uint8_t py = static_cast<int>(pos[IY] / oct_size) % 2;
        uint8_t pz = static_cast<int>(pos[IZ] / oct_size) % 2;

        data_host(offset(iOct)) = {
            iOct, iOct_src,
            1, {px, py, pz},
            false
          };
      }
      else // Copied octant
      {
        assert( mapper.size() == 1);

        uint32_t iOct_src = mapper[0];
        assert( !isghost[0] );
        data_host(offset(iOct)) = {
            iOct, iOct_src,
            0
        };
      }
    });

    Kokkos::deep_copy(data, data_host);
  }

  /// Get Number of mappings (octant pairs)
  KOKKOS_INLINE_FUNCTION
  uint32_t size() const
  {
    return data.size();
  }

  /// Get Number of mappings (octant pairs)
  KOKKOS_INLINE_FUNCTION
  uint32_t getNumOctants() const
  {
    return this->nbOcts;
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
  uint32_t nbOcts;
};

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
                ConfigMap configMap,
                blockSize_t blockSizes,
                DataArrayBlock Usrc,
                DataArrayBlock Usrc_ghost,
                DataArrayBlock& Udest  )
{
  uint32_t nbOcts = remap.getNumOctants();
  uint32_t nbFields = Usrc.extent(1);
  uint32_t nbCellsPerOct = Usrc.extent(0);

  Udest = DataArrayBlock("U",  nbCellsPerOct, nbFields, nbOcts);

  const FunctorData d{
    nbFields,
    Usrc,
    Usrc_ghost,
    Udest,
    blockSizes[IX], blockSizes[IY], blockSizes[IZ],
    ndim
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

} //namespace

void MapUserDataFunctor::apply( const LightOctree_hashmap& lmesh_old,
                                const LightOctree_hashmap& lmesh_new,
                                ConfigMap configMap,
                                blockSize_t blockSizes,
                                DataArrayBlock Usrc,
                                DataArrayBlock Usrc_ghost,
                                DataArrayBlock& Udest  )
{
  apply_aux( AMR_Remapper(lmesh_old, lmesh_new), lmesh_new.getNdim(),
             configMap, blockSizes, Usrc, Usrc_ghost, Udest );
}

void MapUserDataFunctor::apply( const LightOctree_pablo& lmesh_old,
                                const LightOctree_pablo& lmesh_new,
                                ConfigMap configMap,
                                blockSize_t blockSizes,
                                DataArrayBlock Usrc,
                                DataArrayBlock Usrc_ghost,
                                DataArrayBlock& Udest  )
{
  apply_aux( AMR_Remapper(lmesh_new.getMesh()), lmesh_new.getNdim(),
             configMap, blockSizes, Usrc, Usrc_ghost, Udest );
}

} // namespace muscl_block
} // namespace dyablo