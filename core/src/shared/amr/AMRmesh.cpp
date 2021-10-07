#include "AMRmesh.h"

#include "LightOctree.h"

namespace dyablo{

template<>
AMRmesh::AMRmesh_impl( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max)
  : AMRmesh::Impl_t(dim, balance_codim, periodic, level_min, level_max), level_min(level_min), level_max(level_max)
{}

template<>
AMRmesh::~AMRmesh_impl()
{}

template<>
void AMRmesh::updateLightOctree()
{ 
  // Update LightOctree if needed
  if( !lmesh_uptodate )
  {
    lmesh = nullptr;
    lmesh = std::make_unique<LightOctree>( this, level_min, level_max );
    lmesh_uptodate = true;
  }
}

} //namespace dyablo