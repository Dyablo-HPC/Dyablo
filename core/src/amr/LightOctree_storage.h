#pragma once

#include <cassert>
#include <utility>

#include "amr/LightOctree_base.h"
#include "enums.h"

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
   : LightOctree_storage( storage.getNdim(), storage.getNumOctants(), storage.getNumGhosts() )
  {
    Kokkos::deep_copy( this->oct_data, storage.oct_data );
  }

private:
  
  template< class... >
  using void_t = void;
  /// Type-trait to detect if type has a getStorage() method
  template <typename T, typename = void> 
  struct HasStorage : public std::false_type
  {};
  template <typename T>
  struct HasStorage<T, void_t<decltype(std::declval<T>().getStorage())>> : public std::true_type
  {};

public:
  /**
   * Create LightOctree_storage from AMRmesh, when AMRmesh is derived from AMRmesh_impl<>
   * Undelying mesh type must be extracted with getMesh()
   **/
  template< typename AMRmesh_t, typename std::enable_if< HasStorage<typename AMRmesh_t::Impl_t>::value, int >::type = 0 >
  LightOctree_storage(const AMRmesh_t& pmesh)
  : LightOctree_storage( pmesh.getMesh().getStorage() )
  {}
  
  /**
   * Create LightOctree_storage from AMRmesh, when AMRmesh is from LightOctree_storage
   * LightOctree_storage<...> AMRmesh_t::getStorage() is detected and used to copy-construct new storage
   **/
  template< typename AMRmesh_t, typename std::enable_if< HasStorage<AMRmesh_t>::value, int >::type = 0>
  LightOctree_storage(const AMRmesh_t& pmesh)
  : LightOctree_storage( pmesh.getStorage() )
  {}

  /**
   * Create LightOctree_storage from AMRmesh, when AMRmesh is not built around LightOctree_storage
   * Uses AMRmesh public interface to extract mesh
   **/
  template< typename AMRmesh_t, typename std::enable_if< (!HasStorage<AMRmesh_t>::value && !HasStorage<typename AMRmesh_t::Impl_t>::value), int >::type = 0 >
  LightOctree_storage(const AMRmesh_t& pmesh)
  : LightOctree_storage( pmesh.getDim(), pmesh.getNumOctants(), pmesh.getNumGhosts() )
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

        logical_coord_t octant_count = cell_count( level );
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
  LightOctree_storage( int ndim, uint32_t numOctants, uint32_t numGhosts )
  : ndim(ndim), numOctants(numOctants), numGhosts(numGhosts),
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
      real_t oct_size = getSize(iOct);
      return {
          pos[IX] + oct_size/2,
          pos[IY] + oct_size/2,
          pos[IZ] + (ndim-2)*(oct_size/2)
      };
  }
  //! @copydoc LightOctree_base::getCorner()
  KOKKOS_INLINE_FUNCTION 
  pos_t getCorner(const OctantIndex& iOct)  const
  {
      auto lp = get_logical_coords(iOct);
      real_t size = getSize(iOct);
      return {
        lp[IX] * size,
        lp[IY] * size,
        lp[IZ] * size
      };
  }
   //! @copydoc LightOctree_base::getBound()
  KOKKOS_INLINE_FUNCTION 
  bool getBound(const OctantIndex& iOct)  const
  {
    auto lp = get_logical_coords(iOct);
    level_t level = getLevel(iOct);
    logical_coord_t last_oct = cell_count(level)-1;

    return lp[IX] == 0 || lp[IY] == 0 || lp[IZ] == 0 
        || lp[IX] == last_oct || lp[IY] == last_oct || lp[IZ] == last_oct;
  }
  //! @copydoc LightOctree_base::getSize()
  KOKKOS_INLINE_FUNCTION real_t getSize(const OctantIndex& iOct)  const
  {
      return 1.0/cell_count( getLevel(iOct) );
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

  KOKKOS_INLINE_FUNCTION
  static logical_coord_t cell_count( level_t n )
  {
      assert( n < sizeof(logical_coord_t)*8 ); // 
      return (logical_coord_t)1 << n;
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