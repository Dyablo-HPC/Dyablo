#pragma once

#include "AMRmesh_pablo.h"
#include "AMRmesh_hashmap.h"

#include "shared/amr/LightOctree_forward.h"

//#define DYABLO_USE_GPU_MESH

namespace dyablo{

template < typename Impl >
class AMRmesh_impl : private Impl{
private: 
  std::unique_ptr<LightOctree> lmesh;
  uint8_t level_min, level_max;
  bool lmesh_uptodate = false;
public:
  template< typename T, int N >
  using array_t = std::array<T,N>;

  /**
   * Construct a new empty AMR mesh
   * @param dim number of dimensions 2D/3D
   * @param balance_codim 2:1 balance behavior : 
   *               1 ==> balance through faces, 
   *               2 ==> balance through faces and corner
   *               3 ==> balance through faces, edges and corner (3D only)
   * @param periodic set perodicity for each dimension (last is ignored in 2D)
   * @param level_min minimum refinement level 
   * @param level_max maximum refinement level
   * (TODO : clarify level_min/level_max) 
   * Note : Right after construction Mesh has 1 octant
   **/
  AMRmesh_impl( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max)
   : Impl(dim, balance_codim, periodic, level_min, level_max), level_min(level_min), level_max(level_max)
  {}

  //----- Mesh parameters -----
  /// Get number of dimensions
  uint8_t getDim() const
  { return Impl::getDim(); }

  /// Get periodicity of faces {X-,X+,Y-,Y+,[Z-],[Z+]}
  array_t<bool, 6> getPeriodic() const
  { return Impl::getPeriodic(); }

  /// Get periodicity of face i (equivalent to getPeriodic()[i])
  bool getPeriodic(uint8_t i) const
  { return Impl::getPeriodic(i); }

  
  /**
   * Get the LightOctree associated to the current AMR mesh
   * May reallocate LightOctree if mesh has been modified
   **/
  const LightOctree& getLightOctree()
  { 
    // Update LightOctree if needed
    updateLightOctree();
    return *lmesh; 
  }
  
  /// Update LightOctree to make sure next call to getLightOctree() will not reallocate
  void updateLightOctree();

  //----- MPI info -----
  /// MPI rank
  int getRank() const
  { return Impl::getRank(); }
  // MPI communicator size
  int getNproc() const
  { return Impl::getNproc(); }
  // MPI communicator
  MPI_Comm getComm() const
  { return Impl::getComm(); }

  //----- Octant count -----
  /// Get number of local octants
  uint32_t getNumOctants() const
  { return Impl::getNumOctants(); }

  /// Get number of ghost octants
  uint32_t getNumGhosts() const
  { return Impl::getNumGhosts(); }

  /// Get total number of octants across all MPI process
  uint32_t getGlobalNumOctants() const
  { return Impl::getGlobalNumOctants(); }

  /// Get the global id associated to local octant idx
  uint32_t getGlobalIdx( uint32_t idx ) const
  { return Impl::getGlobalIdx(idx); }

  //----- Local Octant info -----
  /**
   * Determine if octant idx has a neighbor outside of the simulation domain
   * Note : periodic neighbors are not outside of domain 
   **/
  bool getBound( uint32_t idx ) const
  { return Impl::getBound(idx); }

  /**
   * Get position of the center of the local cell
   * NOTE : positions are in the unit square [0,1]^3
   **/
  array_t<real_t, 3> getCenter( uint32_t idx ) const
  { return Impl::getCenter(idx); }

  /// Get the position of the corner closest to the origin
  array_t<real_t, 3> getCoordinates( uint32_t idx ) const
  { return Impl::getCoordinates(idx); }

  /**
   * Get the width of the cell
   * Note : all cells are square
   **/ 
  real_t getSize( uint32_t idx ) const
  { return Impl::getSize(idx); }

  /// Get the AMR level of the local cell
  uint8_t getLevel( uint32_t idx ) const
  { return Impl::getLevel(idx); }


  //----- Ghost Octant info -----
  /// Get position of the center of a ghost cell
  array_t<real_t, 3> getCenterGhost( uint32_t idx ) const
  { return Impl::getCenterGhost(idx); }  

  /// Get the position of the corner of a ghost cell closest to the origin
  array_t<real_t, 3> getCoordinatesGhost( uint32_t idx ) const
  { return Impl::getCoordinatesGhost(idx); }

  /// Get the width of the ghost cell
  real_t getSizeGhost( uint32_t idx ) const
  { return Impl::getSizeGhost(idx); }

  /// Get the AMR level of the ghost cell
  uint8_t getLevelGhost( uint32_t idx ) const
  { return Impl::getLevelGhost(idx); }

  //----- Mesh modification -----
  /**
   * Change octants distribution to evenly redistribute the load
   * @param compact_levels are the number of levels to keep compact at the bottom of the tree
   *        all suboctants of octants at level (level_max - compact_level) are kept in the same process
   **/
  void loadBalance(uint8_t compact_levels=0)
  { 
    Impl::loadBalance(compact_levels);
    lmesh_uptodate = false; 
  }  

  /**
   * @copydoc AMRmesh_impl::loadBalance(uint8)
   * @param userData Kokkos::View of octant-related data to be redistributed
   *        userData must have layout Kokkos::LayoutLeft and rightmost index must be octants
   *        for each octant iOct that needs to be moved, values from userData(..., iOct) 
   *        are transfered to the new owning mpi rank
   **/
  void loadBalance_userdata( int compact_levels, DataArrayBlock& userData )
  { 
    Impl::loadBalance_userdata(compact_levels, userData); 
    lmesh_uptodate = false; 
  }
  void loadBalance_userdata( int compact_levels, DataArray& userData )
  { 
    Impl::loadBalance_userdata(compact_levels, userData); 
    lmesh_uptodate = false; 
  }

  /**
   * Set marker for refinement 
   * @param marker +1 to mark iOct for refinement, -1 for coarsening
   **/
  void setMarker(uint32_t iOct, int marker)
  { Impl::setMarker(iOct, marker); }
  /**
   * Coarsen and refine octants according to markers set with setMarker()
   * adapt() includes 2:1 balancing in the directions set with `balance_codim` in the constructor
   * NOTE : refining/coarsening octants by more than one level is not supported (yet?)
   * TODO : remove dummy parameter
   **/
  void adapt(bool dummy = true)
  { 
    Impl::adapt(dummy); 
    lmesh_uptodate = false; 
  }
  /// Refine all octants : same as adapt with all octants marked +1
  void adaptGlobalRefine()
  { 
    Impl::adaptGlobalRefine();
    lmesh_uptodate = false; 
  }
  

  bool check21Balance()
  { return Impl::check21Balance(); }
  bool checkToAdapt()
  { return Impl::checkToAdapt(); }

  const Impl& getMesh() const
  {
    return *this;
  }

  Impl& getMesh()
  {
    return *this;
  }
  
  //TODO hide implementation details
  const std::map<int, std::vector<uint32_t>>& getBordersPerProc() const
  { return Impl::getBordersPerProc(); }

  //TODO remove pablo-specific methods
  // template< typename T >
  // void communicate(T& t)
  // { Impl::communicate(t); }
  void computeConnectivity() {}
  // { Impl::computeConnectivity(); }
  void updateConnectivity() {}
  // { Impl::updateConnectivity(); }
  // template< typename T >
  // void loadBalance(T& t, uint8_t level)
  // { Impl::loadBalance(t,level); }
  // bool getIsNewC(uint32_t idx)
  // { return Impl::getIsNewC(idx); }
  // bool getIsNewR(uint32_t idx) const
  // { return Impl::getIsNewR(idx); }
  // void getMapping(uint32_t & idx, std::vector<uint32_t> & mapper, std::vector<bool> & isghost) const
  // { Impl::getMapping(idx, mapper, isghost); }
  //void findNeighbours(uint32_t iOct, uint8_t iface, uint8_t codim , 
  //                    std::vector<uint32_t>& neighbor_iOcts, std::vector<bool>& neighbor_isGhost) const
  //{ return Impl::findNeighbours(iOct, iface, codim, neighbor_iOcts, neighbor_isGhost); }

};

} // namespace dyablo

#include "shared/amr/LightOctree.h"

namespace dyablo {

template< typename Impl>
void AMRmesh_impl<Impl>::updateLightOctree()
{ 
  // Update LightOctree if needed
  if( !lmesh_uptodate )
  {
    lmesh = std::make_unique<LightOctree>( this, level_min, level_max );
    lmesh_uptodate = true;
  }
}

//template class AMRmesh_impl<AMRmesh_hashmap>;
//template class AMRmesh_impl<AMRmesh_pablo>;

#ifdef DYABLO_USE_GPU_MESH
using AMRmesh = AMRmesh_impl<AMRmesh_hashmap>;
#else
using AMRmesh = AMRmesh_impl<AMRmesh_pablo>;
#endif

}// namespace dyablo