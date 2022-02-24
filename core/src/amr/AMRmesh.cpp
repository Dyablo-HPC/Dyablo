#include "AMRmesh.h"

#include "LightOctree.h"

namespace dyablo{

template<typename Impl_t>
AMRmesh_impl<Impl_t>::AMRmesh_impl( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max)
  : Impl_t(dim, balance_codim, periodic, level_min, level_max), level_min(level_min), level_max(level_max)
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

} //namespace dyablo