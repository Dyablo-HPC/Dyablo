#include "AMRmesh.h"

#include "LightOctree.h"

namespace dyablo{

template<typename Impl_t>
AMRmesh_impl<Impl_t>::AMRmesh_impl( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max)
  : AMRmesh_impl(dim, balance_codim, periodic, level_min, level_max, {(1U << level_min), (1U << level_min), (dim==3)?(1U << level_min):1 } )
{}

template<typename Impl_t>
AMRmesh_impl<Impl_t>::AMRmesh_impl( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max, const Kokkos::Array<uint32_t,3>& coarse_grid_size)
  : Impl_t(dim, balance_codim, periodic, level_min, level_max), level_min(level_min), level_max(level_max)
{
  if( ( coarse_grid_size[IX] != (1U << level_min) ) 
   || ( coarse_grid_size[IY] != (1U << level_min) ) 
   || ( coarse_grid_size[IZ] != ((dim==3)?(1U << level_min):1 )) )
  {
    throw std::runtime_error( "This AMRmesh_implementation doesn't support non-square coarse domain" );
  }
}

template<>
AMRmesh_impl<AMRmesh_hashmap_new>::AMRmesh_impl( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max, const Kokkos::Array<uint32_t,3>& coarse_grid_size)
  : AMRmesh_hashmap_new(dim, balance_codim, periodic, level_min, level_max, coarse_grid_size ), level_min(level_min), level_max(level_max)
{}



template<typename Impl_t>
AMRmesh_impl<Impl_t>::~AMRmesh_impl()
{}

template<typename Impl_t>
void AMRmesh_impl<Impl_t>::updateLightOctree()
{ 
  // Update LightOctree if needed
  if( !lmesh_uptodate )
  {
    lmesh = nullptr;
    lmesh = std::make_unique<LightOctree>( &this->getMesh(), level_min, level_max );
    lmesh_uptodate = true;
  }
}

template class AMRmesh_impl<AMRmesh_pablo>;
template class AMRmesh_impl<AMRmesh_hashmap>;
template class AMRmesh_impl<AMRmesh_hashmap_new>;
} //namespace dyablo