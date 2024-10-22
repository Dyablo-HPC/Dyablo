#pragma once

#include <memory>

#include "FieldManager.h"
#include "utils_hydro.h"
#include "amr/AMRmesh.h"
#include "legacy/utils_block.h"
#include "amr/LightOctree.h"
#include "legacy/LegacyDataArray.h"

namespace dyablo
{


/// @brief Kokkos functor to fill ghost cell data of all octants in the group 
class CopyGhostBlockCellDataFunctor
{
public:
  using index_t = uint32_t;
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<index_t>>;

  struct Params{
    BoundaryConditionType boundary_type_xmin, boundary_type_xmax;
    BoundaryConditionType boundary_type_ymin, boundary_type_ymax;
    BoundaryConditionType boundary_type_zmin, boundary_type_zmax;
    GravityType gravity_type;
  };

  CopyGhostBlockCellDataFunctor(LightOctree lmesh,
                                Params params, id2index_t fm,
                                blockSize_t blockSizes, uint32_t ghostWidth,
                                uint32_t nbOctsPerGroup, LegacyDataArray U,
                                DataArrayBlock Ugroup,
                                uint32_t iGroup,
                                InterfaceFlags interface_flags);
  /**
   * @brief Fill ghost cell data of all octants in the group
   * 
   * static method which does it all: create and execute functor 
   * 
   * Ghost cells of current group octants are filled with values 
   * from neighbor octants. Fills faces, edges and corners.
   **/
  static void apply(LightOctree lmesh,
                    Params params, id2index_t fm, blockSize_t blockSizes,
                    uint32_t ghostWidth, uint32_t nbOctsPerGroup,
                    LegacyDataArray U,
                    DataArrayBlock Ugroup, uint32_t iGroup,
                    InterfaceFlags interface_flags);

  // Called with parallel_for inside apply()
  KOKKOS_INLINE_FUNCTION void operator()(team_policy_t::member_type member) const;

public :
  uint32_t nbTeams; //!< number of thread teams
  

  //! general parameters
  BoundaryConditionType bc_min[3], bc_max[3];

  //! field manager
  id2index_t fm, fm_state ;

  //! block sizes without ghosts
  blockSize_t blockSizes;

  //! ghost width
  uint32_t ghostWidth;

  //! block sizes with ghosts
  uint32_t bx_g;
  uint32_t by_g;
  uint32_t bz_g;

  //! physical bounds of octree
  real_t tree_min[3] = {0,0,0};
  real_t tree_max[3] = {1,1,1};

  //! number of octants per group
  uint32_t nbOctsPerGroup;

  LegacyDataArray U;

  //! heavy data - output - local group array of block data (with ghosts)
  DataArrayBlock Ugroup;

  //! id of group of octants to be copied
  uint32_t iGroup;

  //! 2:1 flagging mechanism
  InterfaceFlags interface_flags;

  // should we copy gravity ?
  bool copy_gravity;

  // number of dimensions for gravity
  int ndim;
  
  //! AMR mesh
  LightOctree lmesh;
};


} // namespace dyablo