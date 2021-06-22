#pragma once

namespace dyablo { 
namespace muscl_block {

template< int ndim >
KOKKOS_INLINE_FUNCTION
HydroState3d getHydroState( const PatchArray& U, const CellIndex& iCell )
{
  HydroState3d res;
  res[ID] = U.at(iCell, ID);
  res[IP] = U.at(iCell, IP);
  res[IU] = U.at(iCell, IU);
  res[IV] = U.at(iCell, IV);
  if(ndim==3) res[IW] = U.at(iCell, IW);
  return res;
}

template< int ndim >
KOKKOS_INLINE_FUNCTION
void setHydroState( const PatchArray& U, const CellIndex& iCell, const HydroState3d& state )
{
  U.at(iCell, ID) = state[ID];
  U.at(iCell, IP) = state[IP];
  U.at(iCell, IU) = state[IU];
  U.at(iCell, IV) = state[IV];
  if(ndim==3) U.at(iCell, IW) = state[IW];
}

template< int ndim >
KOKKOS_INLINE_FUNCTION
void copyGhostBlockCellData(const PatchArray& Uin, const CellIndex& iCell_Ugroup,
                            const ForeachCell::Patch& patch, 
                            real_t xmin, real_t ymin, real_t zmin, 
                            real_t xmax, real_t ymax, real_t zmax, 
                            BoundaryConditionType xbound, BoundaryConditionType ybound, BoundaryConditionType zbound, 
                            const PatchArray& Ugroup)
{
  CellIndex iCell_Uin = Uin.convert_index_ghost(iCell_Ugroup);
  int revert_x = 1, revert_y = 1, revert_z = 1;
  if( iCell_Uin.is_boundary() )
  {
    auto cell_center = patch.getCellCenter(iCell_Ugroup);
    auto cell_size = patch.getCellSize(iCell_Ugroup);
    CellIndex::offset_t boundary_offset{};
    if( cell_center[IX] < xmin )
        boundary_offset[IX] = std::floor((cell_center[IX]-xmin)/cell_size[IX]);
    else if( cell_center[IX] > xmax )
        boundary_offset[IX] = std::ceil((cell_center[IX]-xmax)/cell_size[IX]);                
    if( cell_center[IY] < ymin )
        boundary_offset[IY] = std::floor((cell_center[IY]-ymin)/cell_size[IY]);
    else if( cell_center[IY] > ymax )
        boundary_offset[IY] = std::ceil((cell_center[IY]-ymax)/cell_size[IY]);                
    if(ndim == 3)
    {
      if( cell_center[IZ] < zmin )
          boundary_offset[IZ] = std::floor((cell_center[IZ]-zmin)/cell_size[IZ]);
      else if( cell_center[IZ] > zmax )
          boundary_offset[IZ] = std::ceil((cell_center[IZ]-zmax)/cell_size[IZ]);
    }

    CellIndex iCell_Ugroup_inside = iCell_Ugroup.getNeighbor_local({ (int8_t)(-boundary_offset[IX]), (int8_t)(-boundary_offset[IY]), (int8_t)(-boundary_offset[IZ]) });

    CellIndex::offset_t offset_bc{};
    if( xbound == BC_PERIODIC   ) offset_bc[IX] = boundary_offset[IX];
    if( xbound == BC_REFLECTING ) { offset_bc[IX] = -boundary_offset[IX]; revert_x = -1; }
    if( xbound == BC_ABSORBING  ) offset_bc[IX] = 0;
    if( ybound == BC_PERIODIC   ) offset_bc[IY] = boundary_offset[IY];
    if( ybound == BC_REFLECTING ) { offset_bc[IY] = -boundary_offset[IY]; revert_y = -1; }
    if( ybound == BC_ABSORBING  ) offset_bc[IY] = 0;
    if(ndim == 3)
    {
      if( zbound == BC_PERIODIC   ) offset_bc[IZ] = boundary_offset[IZ];
      if( zbound == BC_REFLECTING ) { offset_bc[IZ] = -boundary_offset[IZ]; revert_z = -1; }
      if( zbound == BC_ABSORBING  ) offset_bc[IZ] = 0;
    }

    iCell_Ugroup_inside = iCell_Ugroup_inside.getNeighbor_local(offset_bc);

    iCell_Uin = Uin.convert_index_ghost( iCell_Ugroup_inside );

    assert( !iCell_Uin.is_boundary() );
  }

  assert( iCell_Uin.is_valid() );
  if( iCell_Uin.level_diff() >= 0 ) 
  {
    // Neighbor is bigger or same size : copy the only neighbor cell
    HydroState3d u = getHydroState<ndim>( Uin, iCell_Uin );
    setHydroState<ndim>( Ugroup, iCell_Ugroup, u );
  }
  else if( iCell_Uin.level_diff() == -1 ) 
  {
    // Neighbor is smaller : get mean of siblings
    int dk_count = (ndim==3)?2:1;
    int nbCells = 2*2*(dk_count);
    HydroState3d u {0,0,0,0,0};
    for( int8_t dk=0; dk<dk_count; dk++ )
    for( int8_t dj=0; dj<=1; dj++ )
    for( int8_t di=0; di<=1; di++ )
    {
        CellIndex iCell_ghost = iCell_Uin.getNeighbor_local({di,dj,dk});
        u[ID] += Uin.at(iCell_ghost, ID)/nbCells;
        u[IP] += Uin.at(iCell_ghost, IP)/nbCells;
        u[IU] += Uin.at(iCell_ghost, IU)/nbCells;
        u[IV] += Uin.at(iCell_ghost, IV)/nbCells;
        if(ndim == 3) u[IW] += Uin.at(iCell_ghost, IW)/nbCells;
    }
    setHydroState<ndim>( Ugroup, iCell_Ugroup, u );
  }
  else assert(false); // Should not happen

  Ugroup.at(iCell_Ugroup, IU) *= revert_x;
  Ugroup.at(iCell_Ugroup, IV) *= revert_y;
  if(ndim == 3) Ugroup.at(iCell_Ugroup, IW) *= revert_z;
}

}// namespace dyablo
}// namespace muscl_block