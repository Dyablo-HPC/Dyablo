#ifndef COPY_BOUNDARIES_BLOCK_CELL_DATA_FUNCTOR_H_
#define COPY_BOUNDARIES_BLOCK_CELL_DATA_FUNCTOR_H_


#include "shared/FieldManager.h"
#include "shared/kokkos_shared.h"

// utils hydro
#include "shared/utils_hydro.h"

// utils block
#include "muscl_block/utils_block.h"

#include "UserPolicies.h"

namespace dyablo
{
namespace muscl_block
{

class CopyBoundariesBlockCellDataFunctor
{
private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using index_t = int32_t;
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<index_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  CopyBoundariesBlockCellDataFunctor(std::shared_ptr<AMRmesh>       pmesh,
                                     HydroParams                    params,
                                     id2index_t                     fm,
                                     blockSize_t                    blockSizes,
                                     uint32_t                       ghostWidth,
                                     uint32_t                       nbOctsPerGroup,
                                     DataArrayBlock                 U,
                                     DataArrayBlock                 Ugroup,
                                     uint32_t                       iGroup,
                                     FlagArrayBlock                 Boundary_flags, 
                                     std::shared_ptr<UserPolicies>  boundaryPolicy) :
    pmesh(pmesh), params(params), fm(fm), blockSizes(blockSizes),
    ghostWidth(ghostWidth), nbOctsPerGroup(nbOctsPerGroup),
    U(U), Ugroup(Ugroup), iGroup(iGroup), Boundary_flags(Boundary_flags),
    boundaryPolicy(boundaryPolicy) {
      bx   = blockSizes[IX];
      by   = blockSizes[IY];
      bz   = blockSizes[IZ];
      bx_g = blockSizes[IX] + 2 * ghostWidth;
      by_g = blockSizes[IY] + 2 * ghostWidth;
      bz_g = blockSizes[IZ] + 2 * ghostWidth;

      ndim = (params.dimType == THREE_D ? 3 : 2);

      gamma0 = params.settings.gamma0;
      smallr = params.settings.smallr;
      smallp = params.settings.smallp;
    }

    static void apply(std::shared_ptr<AMRmesh>      pmesh,
                      ConfigMap                     configMap, 
                      HydroParams                   params, 
                      id2index_t                    fm,
                      blockSize_t                   blockSizes,
                      uint32_t                      ghostWidth,
                      uint32_t                      nbOctsPerGroup,
                      DataArrayBlock                U,
                      DataArrayBlock                Ugroup,
                      uint32_t                      iGroup,
                      FlagArrayBlock                Boundary_flags,
                      std::shared_ptr<UserPolicies> boundaryPolicy)                    
  {
    CopyBoundariesBlockCellDataFunctor functor(pmesh, params, fm, blockSizes, 
                                               ghostWidth, nbOctsPerGroup,
                                               U, Ugroup, iGroup, Boundary_flags,
                                               boundaryPolicy);

    // Setting up the team and execution policy
    uint32_t nbTeams_ = configMap.getInteger("amr", "nbTeams", 16);
    functor.setNbTeams(nbTeams_);

    team_policy_t policy(nbTeams_,
                         Kokkos::AUTO());

    // Start the copy
    Kokkos::parallel_for("dyablo::muscl_block::CopyBoundariesBlockCellDataFunctor",
                         policy, functor);                                          
  }


  KOKKOS_INLINE_FUNCTION
  void fill_boundary_2d(uint32_t iOct, uint32_t iOct_g, uint32_t index, FACE_ID face, BoundaryConditionType bc) const {
    // Calculating boundary cell position and real index
    uint32_t bx_g = bx + 2*ghostWidth;
    uint32_t by_g = by + 2*ghostWidth;

    coord_t coord_cur = (face < FACE_BOTTOM ? 
                         index_to_coord(index, ghostWidth, by_g) :
                         index_to_coord(index, bx_g, ghostWidth));

    // Shifting position according to face being filled
    if (face == FACE_RIGHT)
      coord_cur[IX] += bx + ghostWidth;
    if (face == FACE_TOP)
      coord_cur[IY] += by + ghostWidth;
    
    // Calculating physical coordinates -> todo
    real_t coord_phy[3];

    // Calling the right function for each boundary
    HydroState2d Uout;
    if (bc == BC_REFLECTING)
      Uout = boundaryPolicy->fill_reflecting_2d(Ugroup, iOct_g, index, face, coord_cur, coord_phy);
    else if (bc == BC_ABSORBING)
      Uout = boundaryPolicy->fill_absorbing_2d(Ugroup, iOct_g, index, face, coord_cur, coord_phy);
    else if (bc == BC_USERDEF)
      Uout = boundaryPolicy->fill_bc_2d(Ugroup, iOct_g, index, face, coord_cur, coord_phy);

    // Copying
    uint32_t index_out = coord_cur[IX] + (bx_g * coord_cur[IY]);
    Ugroup(index_out, fm[ID], iOct_g) = Uout[ID];
    Ugroup(index_out, fm[IU], iOct_g) = Uout[IU];
    Ugroup(index_out, fm[IV], iOct_g) = Uout[IV];
    Ugroup(index_out, fm[IE], iOct_g) = Uout[IE];
  }

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void functor2d(team_policy_t::member_type member) const
  {
    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_g = member.league_rank();

    // total number of octants
    uint32_t nbOcts = pmesh->getNumOctants();

    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];

    // How many cells to fill for each boundary
    // We include corners
    uint32_t nbXCells = (by+2*ghostWidth)*ghostWidth;
    uint32_t nbYCells = (bx+2*ghostWidth)*ghostWidth;

    while (iOct < iOctNextGroup and iOct < nbOcts)
    {
      uint32_t bflag = Boundary_flags(iOct_g);
      // Kokkos parallel for is separated for each side to avoid side effects when
      // bx != by
      if (bflag & BF_X) {
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, nbXCells),
          KOKKOS_LAMBDA(const index_t index) { 
            if (bflag & BF_XMIN)
              fill_boundary_2d(iOct, iOct_g, index, FACE_LEFT, params.boundary_type_xmin);
            if (bflag & BF_XMAX)
              fill_boundary_2d(iOct, iOct_g, index, FACE_RIGHT, params.boundary_type_xmax);
          }
        ); 
      }

      if (bflag & BF_Y) {
        Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, nbYCells),
          KOKKOS_LAMBDA(const index_t index) { 
            if (bflag & BF_YMIN)
              fill_boundary_2d(iOct, iOct_g, index, FACE_BOTTOM, params.boundary_type_ymin);
            if (bflag & BF_YMAX)
              fill_boundary_2d(iOct, iOct_g, index, FACE_TOP, params.boundary_type_ymax);
          }
        ); 
      }

      // increase current octant location both in U and Ugroup
      iOct += nbTeams;
      iOct_g += nbTeams;

    } // end while iOct inside current group of octants

  }

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const 
  {

    if (params.dimType == TWO_D)
      functor2d(member);

    // if (params.dimType == THREE_D)
    //   functor3d(member);

  } // operator()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams params;

  //! field manager
  id2index_t fm;

  //! block sizes without ghosts
  blockSize_t blockSizes;

  //! ghost width
  uint32_t ghostWidth;

  //! block sizes without ghosts
  uint32_t bx;
  uint32_t by;
  uint32_t bz;

  //! block sizes with    ghosts
  uint32_t bx_g;
  uint32_t by_g;
  uint32_t bz_g;

  //! number of octants per group
  uint32_t nbOctsPerGroup;

  //! heavy data - input - global array of block data in octants own by current MPI process
  DataArrayBlock U;

  //! heavy data - output - local group array of block data (with ghosts)
  DataArrayBlock Ugroup;

  //! id of group of octants to be copied
  uint32_t iGroup;

  //! Flagging if the current block is along one or more boundaries
  FlagArrayBlock Boundary_flags;

  // number of dimensions for gravity
  int ndim;

  // Specific heat ratio
  real_t gamma0;

  // Minimum density and pressure values
  real_t smallr;
  real_t smallp;

  // Boundary policies
  std::shared_ptr<UserPolicies> boundaryPolicy;

}; // end CopyBoundariesBlockCellDataFunctor

} // end namespace dyablo
} // end namespace muscl_block

#endif