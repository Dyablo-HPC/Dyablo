#ifndef COPY_BOUNDARIES_BLOCK_CELL_DATA_FUNCTOR_H_
#define COPY_BOUNDARIES_BLOCK_CELL_DATA_FUNCTOR_H_


#include "shared/FieldManager.h"
#include "shared/kokkos_shared.h"

// utils hydro
#include "shared/utils_hydro.h"

// utils block
#include "muscl_block/utils_block.h"

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

  CopyBoundariesBlockCellDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                                     HydroParams              params,
                                     id2index_t               fm,
                                     blockSize_t              blockSizes,
                                     uint32_t                 ghostWidth,
                                     uint32_t                 nbOctsPerGroup,
                                     DataArrayBlock           U,
                                     DataArrayBlock           Ugroup,
                                     uint32_t                 iGroup,
                                     FlagArrayBlock           Boundary_flags) :
    pmesh(pmesh), params(params), fm(fm), blockSizes(blockSizes),
    ghostWidth(ghostWidth), nbOctsPerGroup(nbOctsPerGroup),
    U(U), Ugroup(Ugroup), iGroup(iGroup), Boundary_flags(Boundary_flags) {
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

    static void apply(std::shared_ptr<AMRmesh> pmesh,
                      ConfigMap configMap, 
                      HydroParams params, 
                      id2index_t fm,
                      blockSize_t blockSizes,
                      uint32_t ghostWidth,
                      uint32_t nbOctsPerGroup,
                      DataArrayBlock U,
                      DataArrayBlock Ugroup,
                      uint32_t iGroup,
                      FlagArrayBlock Boundary_flags)                    
  {
    CopyBoundariesBlockCellDataFunctor functor(pmesh, params, fm, blockSizes, 
                                               ghostWidth, nbOctsPerGroup,
                                               U, Ugroup, iGroup, Boundary_flags);

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
  HydroState2d fill_reflecting_2d(uint32_t iOct, uint32_t iOct_g, uint32_t index, FACE_ID face, coord_t coords, real_t* coords_phy) const {
    coord_t copy_coords = coords;

    // Getting the correct source coordinates
    if (face == FACE_LEFT)
      copy_coords[IX] = 2 * ghostWidth - coords[IX] - 1;
    else if (face == FACE_BOTTOM)
      copy_coords[IY] = 2 * ghostWidth - coords[IY] - 1;
    else if (face == FACE_RIGHT)
      copy_coords[IX] = 2*(bx+ghostWidth) - 1 - coords[IX];
    else if (face == FACE_TOP)
      copy_coords[IY] = 2*(by+ghostWidth) - 1 - coords[IY];

    uint32_t index_copy = copy_coords[IX] + bx_g * copy_coords[IY];

    HydroState2d Uout;
    Uout[ID] = Ugroup(index_copy, fm[ID], iOct_g);
    Uout[IU] = Ugroup(index_copy, fm[IU], iOct_g);
    Uout[IV] = Ugroup(index_copy, fm[IV], iOct_g);
    Uout[IE] = Ugroup(index_copy, fm[IE], iOct_g);

    if (face == FACE_LEFT or face == FACE_RIGHT)
      Uout[IU] *= -1.0;
    if (face == FACE_BOTTOM or face == FACE_TOP)
      Uout[IV] *= -1.0;

    return Uout;
  }

  KOKKOS_INLINE_FUNCTION
  HydroState2d fill_absorbing_2d(uint32_t iOct, uint32_t iOct_g, uint32_t index, FACE_ID face, coord_t coords, real_t* coords_phy) const {
    coord_t copy_coords = coords;
    if (face == FACE_LEFT)
      copy_coords[IX] = ghostWidth;
    else if (face == FACE_RIGHT)
      copy_coords[IX] = bx_g - ghostWidth - 1;
    else if (face == FACE_BOTTOM)
      copy_coords[IY] = ghostWidth;
    else if (face == FACE_TOP)
      copy_coords[IY] = by_g - ghostWidth - 1;

    uint32_t index_copy = copy_coords[IX] + bx_g * copy_coords[IY];

    //std::cerr << "index = " << index << "; face = " << (int)face << "; coords = " << coords[IX] << " " << coords[IY] << "; Copy coords = " << copy_coords[IX] << " " << copy_coords[IY] << std::endl;

    HydroState2d Uout;
    Uout[ID] = Ugroup(index_copy, fm[ID], iOct_g);
    Uout[IU] = Ugroup(index_copy, fm[IU], iOct_g);
    Uout[IV] = Ugroup(index_copy, fm[IV], iOct_g);
    Uout[IE] = Ugroup(index_copy, fm[IE], iOct_g);

    return Uout;
  }

  KOKKOS_INLINE_FUNCTION
  HydroState2d fill_userdef_2d(uint32_t iOct, uint32_t iOct_g, uint32_t index, FACE_ID face, coord_t coords, real_t* coords_phy) const {

    return HydroState2d{0.0, 0.0, 0.0, 0.0};
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
    
    //std::cerr << "Face : " << (int)face << "; Coords = " << coord_cur[IX] << " " << coord_cur[IY] << std::endl;
    //std::cerr << "Index = " << index << "; iOct = " << iOct << "; iOct_g = " << iOct_g << std::endl; 
    // Calculating physical coordinates -> todo
    real_t coord_phy[3];

    // Calling the right function for each boundary
    // TODO : Replace this by BoundaryPolicy calls
    HydroState2d Uout;
    if (bc == BC_REFLECTING)
      Uout = fill_reflecting_2d(iOct, iOct_g, index, face, coord_cur, coord_phy);
    else if (bc == BC_ABSORBING)
      Uout = fill_absorbing_2d(iOct, iOct_g, index, face, coord_cur, coord_phy);
    else if (bc == BC_USERDEF)
      Uout = fill_userdef_2d(iOct, iOct_g, index, face, coord_cur, coord_phy);

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

}; // end CopyBoundariesBlockCellDataFunctor

} // end namespace dyablo
} // end namespace muscl_block

#endif