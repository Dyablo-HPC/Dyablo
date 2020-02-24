/**
 * \file CopyCornerBlockCellData.h
 * \author Maxime Delorme
 */
#ifndef COPY_CORNER_BLOCK_CELL_DATA_FUNCTOR_H_
#define COPY_CORNER_BLOCK_CELL_DATA_FUNCTOR_H_

#include "shared/FieldManager.h"
#include "shared/kokkos_shared.h"
#include "shared/utils_hydro.h"
#include "muscl_block/utils_block.h"

namespace dyablo {
namespace muscl_block {

  /***************************************/
  /**
   * TODO : Documentation
   **/

class CopyCornerBlockCellDataFunctor {
private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using index_t = int32_t;
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<index_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  CopyCornerBlockCellDataFunctor(std::shared_ptr<AMRmesh> pmesh,
				 HydroParams params,
				 id2index_t fm,
				 blockSize_t blockSizes,
				 uint32_t ghostWidth,
				 uint32_t nbOctsPerGroup,
				 DataArrayBlock U,
				 DataArrayBlock U_ghost,
				 DataArrayBlock Ugroup,
				 uint32_t iGroup) :
    pmesh(pmesh),
    params(params),
    fm(fm),
    blockSizes(blockSizes),
    ghostWidth(ghostWidth),
    nbOctsPerGroup(nbOctsPerGroup),
    U(U),
    U_ghost(U_ghost),
    Ugroup(Ugroup),
    iGroup(iGroup)
  {
    bx   = blockSizes[IX];
    by   = blockSizes[IY];
    bz   = blockSizes[IZ];
    
    bx_g = blockSizes[IX] + 2 * ghostWidth;
    by_g = blockSizes[IY] + 2 * ghostWidth;
    bz_g = blockSizes[IZ] + 2 * ghostWidth;
  }

  // static method that does everything: creates and executes the functor
    static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap configMap, 
                    HydroParams params, 
                    id2index_t fm,
                    blockSize_t blockSizes,
                    uint32_t ghostWidth,
                    uint32_t nbOctsPerGroup,
                    DataArrayBlock U,
                    DataArrayBlock U_ghost,
                    DataArrayBlock Ugroup,
                    uint32_t iGroup)
  {

    CopyCornerBlockCellDataFunctor functor(pmesh, params, fm, 
					   blockSizes, ghostWidth,
					   nbOctsPerGroup, 
					   U, U_ghost, 
					   Ugroup, iGroup);

    /*
     * using kokkos team execution policy
     */
    uint32_t nbTeams_ = configMap.getInteger("amr", "nbTeams", 16);
    functor.setNbTeams(nbTeams_);

    // create execution policy
    team_policy_t policy(nbTeams_,
                         Kokkos::AUTO() /* team size chosen by kokkos */);

    // launch computation (parallel kernel)
    Kokkos::parallel_for("dyablo::muscl_block::CopyCornerBlockCellDataFunctor",
                         policy, functor);
  }

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_bc_2d(uint32_t iOct_local, uint32_t index, uint8_t corner) const {
    coord_t coord_ghost = index_to_coord(index, ghostWidth, ghostWidth);
    coord_t coord_cell  = coord_ghost;
    
    // How to solve boundary conditions in corners ?
    if (corner & CORNER_RIGHT) {
      
    }
    else {

    }
    if (corner & CORNER_TOP) {

    }
    else {

    }
  } // fill_ghost_corner_bc_2d

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_larger_2d(uint32_t iOct_local, uint32_t index, uint8_t corner, uint32_t iOct_neigh,
				   bool isGhost) const {

  } // fill_ghost_corner_larger_2d

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_smaller_2d(uint32_t iOct_local, uint32_t index, uint8_t corner, uint32_t iOct_neigh,
				    bool isGhost) const {

    // Pre-computed offsets for shifting coordinates in the right reference frame
    // x/y_offsets are for the neighbour cell
    // gx/gy_offsets aree for the ghosted block
    uint32_t x_offset  = bx-ghostWidth*2;
    uint32_t y_offset  = by-ghostWidth*2;
    uint32_t gx_offset = bx+ghostWidth;
    uint32_t gy_offset = by+ghostWidth;

    // Coordinates in the neighbour and in the ghosted block
    coord_t coord_neigh = index_to_coord(index, ghostWidth, ghostWidth);
    coord_t coord_ghost = coord_neigh;

    // We multiply by two to shift to smaller cells
    coord_neigh[IX] *= 2;
    coord_neigh[IY] *= 2;

    // Shifting the positions to the correct ids
    if (corner & CORNER_RIGHT) // Right side
      coord_ghost[IX] += gx_offset;
    else                       // Left side
      coord_neigh[IX] += x_offset;
    if (corner & CORNER_TOP)   // Top side
      coord_ghost[IY] += gy_offset;
    else
      coord_neigh[IY] += y_offset;

    // We accumulate the results in this variable
    HydroState2d u = {0.0, 0.0, 0.0, 0.0};

    uint32_t ghost_index = coord_ghost[IX] + bx_g*coord_ghost[IY];

    //std::cerr << ghost_index << " (" << coord_ghost[IX] << "; " << coord_ghost[IY] << ") is receiving from :" << std::endl;

    // Summing all the sub-cells
    for (int ix=0; ix < 2; ++ix) {
      for (int iy=0; iy < 2; ++iy) {
	coord_t coord_sub = coord_neigh;
	coord_sub[IX] += ix;
	coord_sub[IY] += iy;

	uint32_t neigh_index = coord_sub[IX] + bx*coord_sub[IY];
	//	std::cerr << " . " << neigh_index << " (" << coord_sub[IX] << "; " << coord_sub[IY] << ")" << std::endl;

	if (isGhost) {
	  u[ID] += U_ghost(neigh_index, fm[ID], iOct_neigh);
	  u[IU] += U_ghost(neigh_index, fm[IU], iOct_neigh);
	  u[IV] += U_ghost(neigh_index, fm[IV], iOct_neigh);
	  u[IP] += U_ghost(neigh_index, fm[IP], iOct_neigh);
	}
	else {
	  u[ID] += U(neigh_index, fm[ID], iOct_neigh);
	  u[IU] += U(neigh_index, fm[IU], iOct_neigh);
	  u[IV] += U(neigh_index, fm[IV], iOct_neigh);
	  u[IP] += U(neigh_index, fm[IP], iOct_neigh);
	} // end isGhost
      } // end for iy
    } // end for ix

    // And averaging
    Ugroup(ghost_index, fm[ID], iOct_local) = 0.25 * u[ID];
    Ugroup(ghost_index, fm[IU], iOct_local) = 0.25 * u[IU];
    Ugroup(ghost_index, fm[IV], iOct_local) = 0.25 * u[IV];
    Ugroup(ghost_index, fm[IP], iOct_local) = 0.25 * u[IP];

  } // fill_ghost_corner_smaller_2d

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_same_size_2d(uint32_t iOct_local, uint32_t index, uint8_t corner, uint32_t iOct_neigh,
				      bool isGhost) const  {
    // Pre-computed offsets for shifting coordinates in the right reference frame
    // x/y_offsets are for the neighbour cell
    // gx/gy_offsets aree for the ghosted block
    uint32_t x_offset  = bx-ghostWidth;
    uint32_t y_offset  = by-ghostWidth;
    uint32_t gx_offset = bx+ghostWidth;
    uint32_t gy_offset = by+ghostWidth;

    // Coordinates in the neighbour and in the ghosted block
    coord_t coord_neigh = index_to_coord(index, ghostWidth, ghostWidth);
    coord_t coord_ghost = coord_neigh;

    // Shifting the positions to the correct ids
    if (corner & CORNER_RIGHT) // Right side
      coord_ghost[IX] += gx_offset;
    else                       // Left side
      coord_neigh[IX] += x_offset;
    
    if (corner & CORNER_TOP)   // Top side
      coord_ghost[IY] += gy_offset;
    else                       // Bottom side
      coord_neigh[IY] += y_offset;

    // We convert to linear ids
    uint32_t neigh_index = coord_neigh[IX] + bx*coord_neigh[IY];
    uint32_t ghost_index = coord_ghost[IX] + bx_g*coord_ghost[IY];

    //std::cerr << "copying : " << neigh_index << " (" << coord_neigh[IX] << "; " << coord_neigh[IY] << ") -> "
    //	      << ghost_index << " (" << coord_ghost[IX] << "; " << coord_ghost[IY] << ")" << std::endl;

    // And we copy the data
    if (isGhost) {
      Ugroup(ghost_index, fm[ID], iOct_local) = U_ghost(neigh_index, fm[ID], iOct_neigh);
      Ugroup(ghost_index, fm[IU], iOct_local) = U_ghost(neigh_index, fm[IU], iOct_neigh);
      Ugroup(ghost_index, fm[IV], iOct_local) = U_ghost(neigh_index, fm[IV], iOct_neigh);
      Ugroup(ghost_index, fm[IP], iOct_local) = U_ghost(neigh_index, fm[IP], iOct_neigh);
    }
    else {
      Ugroup(ghost_index, fm[ID], iOct_local) = U(neigh_index, fm[ID], iOct_neigh);
      Ugroup(ghost_index, fm[IU], iOct_local) = U(neigh_index, fm[IU], iOct_neigh);
      Ugroup(ghost_index, fm[IV], iOct_local) = U(neigh_index, fm[IV], iOct_neigh);
      Ugroup(ghost_index, fm[IP], iOct_local) = U(neigh_index, fm[IP], iOct_neigh);
    }

  } // fill_ghost_corner_same_size_2d

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_2d(uint32_t iOct, uint32_t iOct_local, uint32_t index, uint8_t corner) const {
    uint8_t codim = 2;

    std::vector<uint32_t> neigh;
    std::vector<bool> isGhost;

    pmesh->findNeighbours(iOct, corner, codim, neigh, isGhost);

    // We treat boundary conditions differently
    if (neigh.size() == 0)
      fill_ghost_corner_bc_2d(iOct_local, index, corner);
    else if (neigh.size() == 1) {  // If we have a neighbour, we disjunct cases
      uint32_t cur_level, neigh_level;

      cur_level   = pmesh->getLevel(iOct);
      neigh_level = pmesh->getLevel(neigh[0]);

      if (cur_level == neigh_level)     // Same level
	fill_ghost_corner_same_size_2d(iOct_local, index, corner, neigh[0], isGhost[0]);
      else if (cur_level > neigh_level) // Larger neighbour
	fill_ghost_corner_larger_2d(iOct_local, index, corner, neigh[0], isGhost[0]);
      else                              // Smaller neighbour
	fill_ghost_corner_smaller_2d(iOct_local, index, corner, neigh[0], isGhost[0]);
    } // end if neigh.size() == 1
    
  } // fill_ghost_corner_2d
  
  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void functor2d(team_policy_t::member_type member) const {

    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_g = member.league_rank();

    // total number of octants
    uint32_t nbOcts = pmesh->getNumOctants();

    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;

    // Number of cells to fill : ghostWidth^2
    uint32_t nbCells = ghostWidth*ghostWidth;

    while (iOct < iOctNextGroup and iOct < nbOcts) {

      // should we do this here ? retrieving the neighbours and everything will be done multiple times
      Kokkos::parallel_for(
	 Kokkos::TeamVectorRange(member, nbCells),
	 KOKKOS_LAMBDA(const index_t index) {
	   // Compute all four corners
	   fill_ghost_corner_2d(iOct, iOct_g, index, CORNER_TOP_LEFT);
	   fill_ghost_corner_2d(iOct, iOct_g, index, CORNER_TOP_RIGHT);
	   fill_ghost_corner_2d(iOct, iOct_g, index, CORNER_BOTTOM_LEFT);
	   fill_ghost_corner_2d(iOct, iOct_g, index, CORNER_BOTTOM_RIGHT);
	 }); // end parallel_for
	   
      // increase current octant location both in U and Ugroup
      iOct += nbTeams;
      iOct_g += nbTeams;

    } // end while iOct inside current group of octants

  } // functor2d()

  // ==============================================================
  // ==============================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(team_policy_t::member_type member) const {
    if (params.dimType == TWO_D)
      functor2d(member);
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

  //! block sizes
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

  //! heavy data - input - global array of block data in MPI ghost octant
  DataArrayBlock U_ghost;

  //! heavy data - output - local group array of block data (with ghosts)
  DataArrayBlock Ugroup;

  //! id of group of octants to be copied
  uint32_t iGroup;

}; // CopyCornerBlockCellDataFunctor

} // namespace muscl_block
} // namespace dyablo

#endif
