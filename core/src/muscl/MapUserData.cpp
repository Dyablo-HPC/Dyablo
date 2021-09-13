#include "muscl/MapUserData.h"

#include "shared/amr/AMR_Remapper.h"

namespace dyablo
{
namespace muscl
{

namespace{

using AMR_Remapper = dyablo::shared::AMR_Remapper;

/// Data to be accessed inside the Kokkos kernel
struct FunctorData{
  uint32_t nbFields;
  DataArray Usrc;
  DataArray Usrc_ghost;
  DataArray Udest;
  uint8_t ndim;
};

/**
 * Fill cell iCell in destination octant m.iOct_dest with 
 * data from octant m.iOct_src when both octants have same size
 **/
KOKKOS_INLINE_FUNCTION
void fill_cell_same_size( const FunctorData& d, 
                          const AMR_Remapper::OctMapping& m)
{
  for( uint32_t iField=0; iField<d.nbFields; iField++ )
  {
    d.Udest(m.iOct_dest, iField) = d.Usrc(m.iOct_src, iField);
  }
}

/**
 * Fill cell iCell in destination octant iOct_dest with 
 * data from octant iOct_src when new octant is smaller
 * This works even if level difference is > than 1
 **/
KOKKOS_INLINE_FUNCTION
void fill_cell_newRefined(const FunctorData& d, 
                          const AMR_Remapper::OctMapping& m)
{
  for( uint32_t iField=0; iField<d.nbFields; iField++ )
  {
    d.Udest(m.iOct_dest, iField) = d.Usrc(m.iOct_src, iField);
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
                          const AMR_Remapper::OctMapping& m)
{
  uint8_t level_diff = -m.level_diff;

  uint32_t suboctant_count = std::pow( 2, level_diff*d.ndim ); 
  for( uint32_t iField=0; iField<d.nbFields; iField++ )
  {
    // Source can be a ghost when dst has just been coarsened
    // (This can't happen otherwise)
    real_t vsrc;
    if( m.isGhost )
      vsrc = d.Usrc_ghost( m.iOct_src, iField )/suboctant_count; 
    else
      vsrc = d.Usrc( m.iOct_src, iField )/suboctant_count;
    // Atomic is needed here because multiple cubcells accumulate their contribution 
    Kokkos::atomic_add(&d.Udest(m.iOct_dest, iField), vsrc);
  }
}

void apply_aux( const AMR_Remapper& remap,
                uint8_t ndim,
                ConfigMap /*configMap*/,
                DataArray Usrc,
                DataArray Usrc_ghost,
                DataArray& Udest  )
{
  uint32_t nbOcts = remap.getNumOctants();
  uint32_t nbFields = Usrc.extent(1);

  Udest = DataArray("U", nbOcts, nbFields);

  const FunctorData d{
    nbFields,
    Usrc,
    Usrc_ghost,
    Udest,
    ndim
  };

  Kokkos::parallel_for("SolverHydroMuscl::map_userdata_after_adapt",
                       remap.size(),
                       KOKKOS_LAMBDA( uint32_t iPair )
  {
    AMR_Remapper::OctMapping mapping = remap[iPair];

    if( mapping.level_diff == 0 )
    {
      fill_cell_same_size(d, mapping);
    }
    else if( mapping.level_diff < 0 )
    {
      fill_cell_newCoarse(d, mapping);
    }
    else if( mapping.level_diff > 0 )
    {
      fill_cell_newRefined(d, mapping);
    }
    else assert(false);

  });
}

} //namespace

void MapUserDataFunctor::apply( const LightOctree_hashmap& lmesh_old,
                                const LightOctree_hashmap& lmesh_new,
                                ConfigMap configMap,
                                DataArray Usrc,
                                DataArray Usrc_ghost,
                                DataArray& Udest  )
{
  apply_aux( AMR_Remapper(lmesh_old, lmesh_new), lmesh_new.getNdim(),
             configMap, Usrc, Usrc_ghost, Udest );
}

void MapUserDataFunctor::apply( const LightOctree_pablo& lmesh_old,
                                const LightOctree_pablo& lmesh_new,
                                ConfigMap configMap,
                                DataArray Usrc,
                                DataArray Usrc_ghost,
                                DataArray& Udest  )
{
  apply_aux( AMR_Remapper(lmesh_new.getMesh()), lmesh_new.getNdim(),
             configMap, Usrc, Usrc_ghost, Udest );
}

} // namespace muscl
} // namespace dyablo