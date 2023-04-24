#include "AMRmesh.h"

#include "LightOctree.h"
#include "utils/config/ConfigMap.h"

namespace dyablo{

template<typename Impl_t>
AMRmesh_impl<Impl_t>::AMRmesh_impl( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max)
  : AMRmesh_impl(dim, balance_codim, periodic, level_min, level_max, {(1U << level_min), (1U << level_min), (dim==3)?(1U << level_min):1 } )
{}

template<typename Impl_t>
AMRmesh_impl<Impl_t>::~AMRmesh_impl()
{}
template<typename Impl_t>
typename AMRmesh_impl<Impl_t>::Parameters AMRmesh_impl<Impl_t>::parse_parameters(ConfigMap& configMap)
{
  Parameters res;
  int ndim = configMap.getValue<int>("mesh", "ndim", 3);
  res.dim = ndim;
  BoundaryConditionType bxmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING);
  BoundaryConditionType bxmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING);
  BoundaryConditionType bymin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING);
  BoundaryConditionType bymax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING);
  BoundaryConditionType bzmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING);
  BoundaryConditionType bzmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING);
  res.periodic = {
    bxmin == BC_PERIODIC || bxmax == BC_PERIODIC,
    bymin == BC_PERIODIC || bymax == BC_PERIODIC,
    bzmin == BC_PERIODIC || bzmax == BC_PERIODIC
  };

  if( configMap.hasValue( "amr","level_min" ) )
  {
    res.level_min = configMap.getValue<int>("amr","level_min");
    res.level_max = configMap.getValue<int>("amr","level_max", res.level_min + 3);
    uint32_t max_width = (1U << res.level_min);
    res.coarse_grid_size = { 
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_x", max_width ),
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_y", max_width ),
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_z", (ndim==3)?max_width:1 ) 
    };

    DYABLO_ASSERT_HOST_RELEASE( res.coarse_grid_size[IX] <= max_width, 
      "amr/coarse_oct_resolution_x (" << res.coarse_grid_size[IX] << ") is too big for level_min (" << res.level_min << "), max allowed is " << max_width );
    DYABLO_ASSERT_HOST_RELEASE( res.coarse_grid_size[IY] <= max_width, 
      "amr/coarse_oct_resolution_y (" << res.coarse_grid_size[IY] << ") is too big for level_min (" << res.level_min << "), max allowed is " << max_width );
    DYABLO_ASSERT_HOST_RELEASE( res.coarse_grid_size[IZ] <= ((ndim==3)?max_width:1), 
      "amr/coarse_oct_resolution_z (" << res.coarse_grid_size[IZ] << ") is too big for level_min (" << res.level_min << "), max allowed is " << ((ndim==3)?max_width:1) );
  
  }
  else if( configMap.hasValue("amr","coarse_oct_resolution_x") 
        && configMap.hasValue("amr","coarse_oct_resolution_y") 
        && ( ( ndim == 2 ) || configMap.hasValue("amr","coarse_oct_resolution_z") ) )
  {
    res.coarse_grid_size = { 
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_x" ),
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_y" ),
        configMap.getValue<uint32_t>("amr","coarse_oct_resolution_z", 1 ) 
    };

    uint32_t width = std::max({res.coarse_grid_size[IX], res.coarse_grid_size[IY], res.coarse_grid_size[IZ]} );
    res.level_min = configMap.getValue<int>( "amr", "level_min", std::ceil(std::log2( width ))  );
    res.level_max = configMap.getValue<int>( "amr", "level_max", res.level_min + 3);
  }

  return res;
}

template<typename Impl_t>
void AMRmesh_impl<Impl_t>::updateLightOctree()
{ 
  // Update LightOctree if needed
  if( !lmesh_uptodate() )
  {
    lmesh = nullptr;
    lmesh = std::make_unique<LightOctree>( &this->getMesh(), level_min, level_max );
    this->lmesh_epoch = Impl_t::pmesh_epoch;
  }
}

template<typename Impl_t>
const LightOctree& AMRmesh_impl<Impl_t>::getLightOctree()
{ 
  // Update LightOctree if needed
  updateLightOctree();
  DYABLO_ASSERT_HOST_RELEASE( lmesh->getNumOctants() == this->getNumOctants(), "LightOctree::getLightOctree() is outdated pmesh " << this->getNumOctants() << "octs vs lmesh " << lmesh->getNumOctants() << "octs" );
  DYABLO_ASSERT_HOST_RELEASE( lmesh->getNumGhosts() == this->getNumGhosts(), "LightOctree::getLightOctree() is outdated pmesh " << this->getNumGhosts() << "ghosts vs lmesh " << lmesh->getNumGhosts() << "ghosts" );
  return *lmesh; 
}

#ifdef DYABLO_COMPILE_PABLO
  template class AMRmesh_impl<AMRmesh_pablo>;
#endif
template class AMRmesh_impl<AMRmesh_hashmap>;
template class AMRmesh_impl<AMRmesh_hashmap_new>;
} //namespace dyablo
