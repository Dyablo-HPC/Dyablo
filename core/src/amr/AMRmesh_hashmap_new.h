#pragma once

#include "utils/mpi/GlobalMpiSession.h"
#include "amr/LightOctree_storage.h"
#include "kokkos_shared.h"

namespace dyablo {

class AMRmesh_hashmap_new
{
public:
  using Storage_t = LightOctree_storage< Kokkos::DefaultHostExecutionSpace::memory_space >;
  using logical_coord_t = Storage_t::logical_coord_t;
  using level_t = Storage_t::level_t;
  using oct_index_t = uint32_t;
  using global_oct_index_t = uint64_t;
  template< typename T, int N >
  using array_t = std::array<T,N>;

  AMRmesh_hashmap_new( int dim, int balance_codim, 
                      const std::array<bool,3>& periodic, 
                      uint8_t level_min, uint8_t level_max,
                      const MpiComm& mpi_comm = GlobalMpiSession::get_comm_world());

  ~AMRmesh_hashmap_new();

  const Storage_t& getStorage() const
  {
    return storage;
  }

  oct_index_t getNumOctants() const
  { return storage.getNumOctants(); }
  oct_index_t getNumGhosts() const
  { return storage.getNumGhosts(); }

  uint8_t getDim() const
  { return storage.getNdim(); }

  bool getPeriodic( int i ) const
  { 
    return {periodic[i/2]}; 
  }
  int get_max_supported_level()
  {
    return 20; // Maybe more? (But never tested)
  }

  const MpiComm& getMpiComm() const
  { return mpi_comm; }

  global_oct_index_t getGlobalNumOctants() const
  { return total_num_octs; }

  global_oct_index_t getGlobalIdx( oct_index_t idx ) const
  { return first_local_oct + idx; }

  bool getBound( oct_index_t idx ) const
  { return storage.getBound( {idx, false} ); }

  array_t<real_t, 3> getCenter( uint32_t idx ) const
  { 
    auto p = storage.getCenter( {idx, false} );
    return {p[IX], p[IY], p[IZ]};
  }
  
  array_t<real_t, 3> getCenterGhost( uint32_t idx ) const
  { 
    auto p = storage.getCenter( {idx, true} );
    return {p[IX], p[IY], p[IZ]};
  }

  array_t<real_t, 3> getCoordinates( uint32_t idx ) const
  { 
    auto p = storage.getCorner( {idx, false} );
    return {p[IX], p[IY], p[IZ]};
  }
  
  array_t<real_t, 3> getCoordinatesGhost( uint32_t idx ) const
  { 
    auto p = storage.getCorner( {idx, true} );
    return {p[IX], p[IY], p[IZ]};
  }

  real_t getSize( uint32_t idx ) const
  { return storage.getSize( {idx, false} ); }

  real_t getSizeGhost( uint32_t idx ) const
  { return storage.getSize( {idx, true} ); }

  level_t getLevel( uint32_t idx ) const
  { return storage.getLevel( {idx, false} ); }

  level_t getLevelGhost( uint32_t idx ) const
  { return storage.getLevel( {idx, true} ); }

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
  Storage_t storage;

  Kokkos::Array<bool,3> periodic;
  MpiComm mpi_comm;
  global_oct_index_t total_num_octs, first_local_oct;

  // Pimpl idiom
  struct PData;
  std::unique_ptr<PData> pdata;  
};

}// namespace dyablo