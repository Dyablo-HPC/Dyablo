#pragma once

#include <cassert>
#include <utility>
#include "utils/misc/Dyablo_assert.h"
#include "amr/LightOctree_base.h"
#include "enums.h"

namespace LightOctree_storage_impl{

/// When T is AMRmesh_impl<AMRmesh_*> where  AMRmesh_* has a getStorage() method, use this
template< typename T >
static decltype(std::declval<T>().getMesh().getStorage()) getStorage( const T& t ) 
{ return t.getMesh().getStorage(); }
/// When T is T has a getStorage() method, use it
template< typename T >
static decltype(std::declval<T>().getStorage()) getStorage( const T& t ) 
{ return t.getStorage(); }

template< class... >
using void_t = void;
/// Type-trait to detect if T is compatible with getStorage()
template <typename T, typename = void> 
struct HasStorage : public std::false_type
{};
template <typename T>
struct HasStorage<T, void_t<decltype(getStorage(std::declval<T>()))>> : public std::true_type
{};

} // namespace LightOctree_storage_impl

namespace dyablo { 

template< typename MemorySpace_ = Kokkos::View<int*>::memory_space >
class LightOctree_storage
{
public:
  using MemorySpace = MemorySpace_;
  using logical_coord_t = uint32_t;
  using level_t = logical_coord_t;
  using oct_data_t = Kokkos::View< logical_coord_t**, Kokkos::LayoutLeft, MemorySpace >;
  using pos_t = Kokkos::Array<real_t,3>;
  using coarse_grid_size_t = Kokkos::Array<logical_coord_t,3>;
protected:
  using OctantIndex = LightOctree_base::OctantIndex;
  //! Index to access different fields in `oct_data`
  enum oct_data_field_t{
      ICORNERX, 
      ICORNERY, 
      ICORNERZ, 
      ILEVEL,
      OCT_DATA_COUNT
  };

public:
  LightOctree_storage() = default;
  LightOctree_storage(const LightOctree_storage& lmesh) = default;
  LightOctree_storage& operator=(const LightOctree_storage& lmesh) = default;
  LightOctree_storage(LightOctree_storage&& lmesh) = default;
  LightOctree_storage& operator=(LightOctree_storage&& lmesh) = default;

  template< typename MemorySpace_t >
  LightOctree_storage(const LightOctree_storage<MemorySpace_t>& storage)
   : LightOctree_storage( storage.getNdim(), storage.getNumOctants(), storage.getNumGhosts(), storage.level_min, storage.coarse_grid_size )
  {
    Kokkos::deep_copy( this->oct_data, storage.oct_data );
  }

public:
  /**
   * Create LightOctree_storage from AMRmesh, when AMRmesh_t uses LightOctree_storage
   * LightOctree_storage<...> AMRmesh_t::getStorage() is detected and used to copy-construct new storage
   **/
  template< typename AMRmesh_t, typename std::enable_if< LightOctree_storage_impl::HasStorage<AMRmesh_t>::value, int >::type = 0>
  LightOctree_storage(const AMRmesh_t& pmesh)
  : LightOctree_storage( LightOctree_storage_impl::getStorage(pmesh) )
  {}

  /**
   * Create LightOctree_storage from AMRmesh, when AMRmesh is not built around LightOctree_storage
   * Uses AMRmesh public interface to extract mesh
   **/
  template< typename AMRmesh_t, typename std::enable_if< !LightOctree_storage_impl::HasStorage<AMRmesh_t>::value, int >::type = 0 >
  LightOctree_storage(const AMRmesh_t& pmesh)
  : LightOctree_storage( pmesh.getDim(), pmesh.getNumOctants(), pmesh.getNumGhosts(), pmesh.get_level_min() )
  {
    private_init(pmesh);   
  }

  /**
   * DO NOT CALL THIS YOURSELF
   * this is here because KOKKOS_LAMBDAS cannot be declared in constructors or private methods
   **/
  template< typename AMRmesh_t >
  void private_init(const AMRmesh_t& pmesh)
  {
    typename oct_data_t::HostMirror oct_data_host = Kokkos::create_mirror_view(oct_data);

    Kokkos::parallel_for( "LightOctree_storage::copydata", 
                        Kokkos::RangePolicy<Kokkos::OpenMP>(0, numOctants+numGhosts),
                        [&]( uint32_t ioct_local )
    {
        OctantIndex oct = OctantIndex::iOctLocal_to_OctantIndex( ioct_local, numOctants );

        auto c = oct.isGhost ? 
                pmesh.getCoordinatesGhost(oct.iOct):
                pmesh.getCoordinates(oct.iOct);
        uint8_t level = oct.isGhost ? 
                pmesh.getLevelGhost(oct.iOct):
                pmesh.getLevel(oct.iOct);

        logical_coord_t octant_count =( 1U << level );
        real_t octant_size = 1.0/octant_count;
        oct_data_host(ioct_local, ICORNERX) = std::floor(c[IX]/octant_size);
        oct_data_host(ioct_local, ICORNERY) = std::floor(c[IY]/octant_size);
        oct_data_host(ioct_local, ICORNERZ) = (ndim-2)*std::floor(c[IZ]/octant_size);
        oct_data_host(ioct_local, ILEVEL) = level;
    });

    // Copy data to device
    Kokkos::deep_copy(oct_data,oct_data_host);
  }


  // Create an empty LightOctree_storage
  LightOctree_storage( int ndim, uint32_t numOctants, uint32_t numGhosts, level_t level_min, const coarse_grid_size_t& coarse_grid_size )
  : ndim(ndim), numOctants(numOctants), numGhosts(numGhosts), level_min(level_min),
    coarse_grid_size(coarse_grid_size),
    oct_data("LightOctree_storage", numOctants+numGhosts, oct_data_field_t::OCT_DATA_COUNT)
  {}

  // Create an empty LightOctree_storage (full coarse grid version)
  LightOctree_storage( int ndim, uint32_t numOctants, uint32_t numGhosts, level_t level_min)
  : ndim(ndim), numOctants(numOctants), numGhosts(numGhosts), level_min(level_min),
    coarse_grid_size( { (1U << level_min), (1U << level_min), (ndim==3)?(1U << level_min):1 }  ),
    oct_data("LightOctree_storage", numOctants+numGhosts, oct_data_field_t::OCT_DATA_COUNT)
  {}

public:
  //! @copydoc LightOctree_base::getNumOctants()
  KOKKOS_INLINE_FUNCTION 
  uint32_t getNumOctants() const
  {
      return numOctants;
  }

  //! @copydoc LightOctree_base::getNumGhosts()
  KOKKOS_INLINE_FUNCTION 
  uint32_t getNumGhosts() const
  {
      return numGhosts;
  }

  //! @copydoc LightOctree_base::getNdim()
  KOKKOS_INLINE_FUNCTION 
  uint8_t getNdim() const
  {
      return ndim;
  }
  //! @copydoc LightOctree_base::getCenter()
  KOKKOS_INLINE_FUNCTION 
  pos_t getCenter(const OctantIndex& iOct)  const
  {
      pos_t pos = getCorner(iOct);
      auto oct_size = getSize(iOct);
      return {
          pos[IX] + oct_size[IX]/2,
          pos[IY] + oct_size[IY]/2,
          pos[IZ] + (ndim-2)*(oct_size[IZ]/2)
      };
  }
  //! @copydoc LightOctree_base::getCorner()
  KOKKOS_INLINE_FUNCTION 
  pos_t getCorner(const OctantIndex& iOct)  const
  {
      auto lp = get_logical_coords(iOct);
      auto size = getSize(iOct);
      return {
        lp[IX] * size[IX],
        lp[IY] * size[IY],
        lp[IZ] * size[IZ]
      };
  }
   //! @copydoc LightOctree_base::getBound()
  KOKKOS_INLINE_FUNCTION 
  bool getBound(const OctantIndex& iOct)  const
  {
    auto lp = get_logical_coords(iOct);
    level_t level = getLevel(iOct);
    logical_coord_t last_oct_x = cell_count(IX, level)-1;
    logical_coord_t last_oct_y = cell_count(IY, level)-1;
    logical_coord_t last_oct_z = cell_count(IZ, level)-1;

    return lp[IX] == 0 || lp[IY] == 0 || lp[IZ] == 0 
        || lp[IX] == last_oct_x || lp[IY] == last_oct_y || lp[IZ] == last_oct_z;
  }
  //! @copydoc LightOctree_base::getSize()
  KOKKOS_INLINE_FUNCTION 
  pos_t getSize(const OctantIndex& iOct)  const
  {
      return { 1.0/cell_count( IX, getLevel(iOct) ),
               1.0/cell_count( IY, getLevel(iOct) ),
               1.0/cell_count( IZ, getLevel(iOct) ) };
  }
  //! @copydoc LightOctree_base::getLevel()
  KOKKOS_INLINE_FUNCTION level_t getLevel(const OctantIndex& iOct)  const
  {
      return oct_data(get_ioct_local(iOct), ILEVEL);
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<logical_coord_t, 3> get_logical_coords( const OctantIndex& iOct )  const
  {
    uint32_t iOct_local = get_ioct_local(iOct);
    return {
        oct_data(iOct_local, ICORNERX),
        oct_data(iOct_local, ICORNERY),
        oct_data(iOct_local, ICORNERZ),
    };
  }

  KOKKOS_INLINE_FUNCTION 
  void set( const OctantIndex& iOct, 
            logical_coord_t ix, logical_coord_t iy, logical_coord_t iz, 
            level_t level ) const
  {
    uint32_t iOct_local = get_ioct_local(iOct);
    oct_data(iOct_local, ICORNERX) = ix;
    oct_data(iOct_local, ICORNERY) = iy;
    oct_data(iOct_local, ICORNERZ) = iz;
    oct_data(iOct_local, ILEVEL) = level;
  }

  oct_data_t getLocalSubview() const
  {
    return Kokkos::subview( 
      oct_data, 
      std::make_pair( (uint32_t)0, getNumOctants() ),
      Kokkos::ALL() );
  }

  oct_data_t getGhostSubview() const
  {
    return Kokkos::subview( 
      oct_data, 
      std::make_pair( getNumOctants(), getNumOctants()+getNumGhosts() ),
      Kokkos::ALL() );
  }

  int ndim;
  uint32_t numOctants, numGhosts; //! Number of local octants (no ghosts), Number of ghosts.
  level_t level_min;
  coarse_grid_size_t coarse_grid_size;

  KOKKOS_INLINE_FUNCTION
  logical_coord_t cell_count( ComponentIndex3D idim, level_t n ) const
  {
      DYABLO_ASSERT_KOKKOS_DEBUG( n>=level_min, "Cannot ask cell_count with level < level_min" );
      DYABLO_ASSERT_KOKKOS_DEBUG( n < sizeof(logical_coord_t)*8, "Overflow : cell_count too big for logical_coord_t" );
      return coarse_grid_size[idim] << (n-level_min);
  }

  //! Kokkos::view containing octants position and level 
  //! ex: (oct_data(iOct, ILEVEL) is octant level)
  oct_data_t oct_data;
  
  KOKKOS_INLINE_FUNCTION 
  uint32_t get_ioct_local(const OctantIndex& oct) const
  {
      // Ghosts are stored after non-ghosts
      return OctantIndex::OctantIndex_to_iOctLocal(oct, numOctants);
  }
};

}