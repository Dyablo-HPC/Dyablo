#pragma once

#include "utils/mpi/GlobalMpiSession.h"
#include "amr/LightOctree_storage.h"
#include "kokkos_shared.h"

namespace dyablo {

class AMRmesh_hashmap_new : public LightOctree_storage< Kokkos::DefaultHostExecutionSpace::memory_space >
{
public:
  using Storage_t = LightOctree_storage< Kokkos::DefaultHostExecutionSpace::memory_space >;
  using level_t = uint16_t;
  using oct_index_t = uint32_t;
  using global_oct_index_t = uint64_t;
  template< typename T, int N >
  using array_t = std::array<T,N>;

  AMRmesh_hashmap_new( int dim, int balance_codim, 
                      const std::array<bool,3>& periodic, 
                      uint8_t level_min, uint8_t level_max,
                      const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world());

  ~AMRmesh_hashmap_new();

  const Storage_t getStorage();

  using Storage_t::getNumOctants;
  using Storage_t::getNumGhosts;

  uint8_t getDim() const
  { return Storage_t::getNdim(); }

  bool getPeriodic( int i ) const
  { 
    return {periodic[i/2]}; 
  }

  const MpiComm& getMpiComm() const
  { return mpi_comm; }

  global_oct_index_t getGlobalNumOctants() const
  { return total_num_octs; }

  global_oct_index_t getGlobalIdx( oct_index_t idx ) const
  { return first_local_oct + idx; }

  bool getBound( oct_index_t idx ) const
  { return Storage_t::getBound( {idx, false} ); }

  array_t<real_t, 3> getCenter( uint32_t idx ) const
  { 
    auto p = Storage_t::getCenter( {idx, false} );
    return {p[IX], p[IY], p[IZ]};
  }
  
  array_t<real_t, 3> getCenterGhost( uint32_t idx ) const
  { 
    auto p = Storage_t::getCenter( {idx, true} );
    return {p[IX], p[IY], p[IZ]};
  }

  array_t<real_t, 3> getCoordinates( uint32_t idx ) const
  { 
    auto p = Storage_t::getCorner( {idx, false} );
    return {p[IX], p[IY], p[IZ]};
  }
  
  array_t<real_t, 3> getCoordinatesGhost( uint32_t idx ) const
  { 
    auto p = Storage_t::getCorner( {idx, true} );
    return {p[IX], p[IY], p[IZ]};
  }

  real_t getSize( uint32_t idx ) const
  { return Storage_t::getSize( {idx, false} ); }

  real_t getSizeGhost( uint32_t idx ) const
  { return Storage_t::getSize( {idx, true} ); }

  level_t getLevel( uint32_t idx ) const
  { return Storage_t::getLevel( {idx, false} ); }

  level_t getLevelGhost( uint32_t idx ) const
  { return Storage_t::getLevel( {idx, true} ); }

  // Output is not used in AMRmesh_impl
  std::map<int, std::vector<uint32_t>> loadBalance( level_t compact_levels );
  void loadBalance_userdata( level_t compact_levels, DataArrayBlock& userData );

  void setMarker(uint32_t iOct, int marker);
  void adapt(bool dummy);
  
  void adaptGlobalRefine();

  const std::map<int, std::vector<uint32_t>>& getBordersPerProc() const;

  bool check21Balance()
  {
      //TODO
      return true;
  }

  bool checkToAdapt()
  {
      //TODO
      return false;
  }

private : 
  Kokkos::Array<bool,3> periodic;
  MpiComm mpi_comm;
  global_oct_index_t total_num_octs, first_local_oct;

  // Pimpl idiom
  struct PData;
  std::unique_ptr<PData> pdata;  
};

}// namespace dyablo