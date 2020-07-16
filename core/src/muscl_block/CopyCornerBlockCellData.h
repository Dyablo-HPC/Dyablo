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
   * This functor copies in a full ghosted block the corners
   * There are several cases to take into account : 
   *   . Same size neighbours are treated in fill_ghost_corner_same_size_Xd
   *   . Larger neighbours are treated in fill_ghost_corner_larger_Xd
   *   . Smaller neighbours are treated in fill_ghost_corner_smaller_Xd
   *   . Face boundary conditions are treated in fill_ghost_corner_bc_face_Xd
   *   . Corner boundary conditions are treated in fill_ghost_corner_bc_corner_Xd
   *
   * \note The copy for face boundary conditions relies on filled ghost faces
   * hence this kernel should strictly be used AFTER CopyFaceBlockCellDataFunctor
   **/

class CopyCornerBlockCellDataFunctor {
private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using offset_t = Kokkos::Array<uint32_t,3>;
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
				 uint32_t iGroup,
				 FlagArrayBlock Interface_flags) :
    pmesh(pmesh),
    params(params),
    fm(fm),
    blockSizes(blockSizes),
    ghostWidth(ghostWidth),
    nbOctsPerGroup(nbOctsPerGroup),
    U(U),
    U_ghost(U_ghost),
    Ugroup(Ugroup),
    iGroup(iGroup),
    Interface_flags(Interface_flags),
    eps(std::numeric_limits<real_t>::epsilon())
  {
    bx   = blockSizes[IX];
    by   = blockSizes[IY];
    bz   = blockSizes[IZ];
    
    bx_g = blockSizes[IX] + 2 * ghostWidth;
    by_g = blockSizes[IY] + 2 * ghostWidth;
    bz_g = blockSizes[IZ] + 2 * ghostWidth;

    copy_gravity = (params.gravity_type & GRAVITY_FIELD);
    ndim = (params.dimType == THREE_D ? 3 : 2);
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
		      uint32_t iGroup,
		      FlagArrayBlock Interface_flags)
  {

    CopyCornerBlockCellDataFunctor functor(pmesh, params, fm, 
					   blockSizes, ghostWidth,
					   nbOctsPerGroup, 
					   U, U_ghost, Ugroup, 
             iGroup, Interface_flags);

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
  /**
   * This routine returns the iOct of the block corresponding to the indicated corner
   * This is trickier than expected since PABLO does not return a corner neighbour
   * in the case where the corner of the block is currently in the middle of a larger
   * face. e.g:
   *      +------+
   *      |      |
   *  +---X      |
   *  |   |      |
   *  +---+------+
   *
   * For the calculation of coordinates, please refer to the design doc "corners_larger_neighbours.pdf" for 
   * a description of the cases referred to in the comments.
   *
   * \param[in] iOct is of the current octant in the global tree
   * \param[in] iOct_local id of the current octant in the active group to be filled
   * \param[in] corner to be filled here
   * \param[out] isBoundary if we're hitting a non-periodic boundary
   * \param[out] neigh the id of the neighbour if we're not at a boundary
   * \param[out] isGhost if the neighbour is in the ghost cells of the current process
   * \param[out] coord_neigh the position in the neighbouring block from where we want to start to copy
   **/
  KOKKOS_INLINE_FUNCTION
  void get_corner_neighbours(uint32_t iOct, uint32_t iOct_local, uint8_t corner, bool &isBoundary, uint32_t &neigh, bool &isGhost, coord_t &coord_neigh) const {
    // Temp struct to communicate with PABLO
    std::vector<uint32_t> neigh_v;
    std::vector<bool> isGhost_v;
    const uint8_t corner_codim = 2;
    const uint8_t face_codim   = 1;
    
    isBoundary = false;

    // If we're not, we check the neighbours using PABLO
    pmesh->findNeighbours(iOct, corner, corner_codim, neigh_v, isGhost_v);

    /*if (iOct == 167 or iOct == 40) {
      std::cerr << "iOct = " << iOct << "; corner = " << (int)corner << "; neigh.size = " << neigh_v.size()
      << std::endl;
    }*/

    // All good, we have a neighbour in PABLO, we return it
    if (neigh_v.size() > 0) {
      neigh      = neigh_v[0];
      isGhost    = isGhost_v[0];

      // Finding coordinates from where to copy
      if (corner & CORNER_RIGHT)  // Right face  -> Cases 1 and 9
	      coord_neigh[IX] = 0;
      else                        // Left face   -> Cases 4 and 12
	      coord_neigh[IX] = bx - ghostWidth/2;

      if (corner & CORNER_TOP)    // Top face    -> Cases 9 and 12
	      coord_neigh[IY] = 0;
      else                        // Bottom face -> Cases 1 and 4
	      coord_neigh[IY] = by - ghostWidth/2;

      return;
    }
    else {
      // Testing for non conformal corners : the corner would then point to the center of a larger edge
      // In that case, PABLO returns no neighbours so we have to detect
      // the neighbour using the edge.
      // These correspond to design cases 2, 3, 5, 6, 7, 8, 10 and 11

      // X interface
      uint8_t iface_x, iface_y;
      InterfaceType nc_flag_x, nc_flag_y;
      if (corner & CORNER_RIGHT) {
	      iface_x   = FACE_RIGHT;
	      nc_flag_x = INTERFACE_XMAX_BIGGER;
      }
      else {
	      iface_x   = FACE_LEFT;
	      nc_flag_x = INTERFACE_XMIN_BIGGER;
      }
      // Y interface
      if (corner & CORNER_TOP) {
        iface_y   = FACE_TOP;
        nc_flag_y = INTERFACE_YMAX_BIGGER;
      }
      else {
        iface_y   = FACE_BOTTOM;
        nc_flag_y = INTERFACE_YMIN_BIGGER;
      }

      // If the corresponding neighbour is bigger, we return it
      std::vector<uint32_t> neigh_x, neigh_y;
      std::vector<bool> isGhost_x, isGhost_y;
      pmesh->findNeighbours(iOct, iface_x, face_codim, neigh_x, isGhost_x);
      pmesh->findNeighbours(iOct, iface_y, face_codim, neigh_y, isGhost_y);

      if (neigh_x.size() == 0 or neigh_y.size() == 0) {
        isBoundary = true;
        return;
      }

      if (Interface_flags(iOct_local) & nc_flag_x) {
        // X interface has a larger neighbour, that means we are in cases 5, 6, 7 or 8
        if (neigh_x.size() > 0) {
          isBoundary = false;
          neigh      = neigh_x[0];
          isGhost    = isGhost_x[0];

          if (corner & CORNER_RIGHT) // Corner on the right side;  Cases 5 and 7
            coord_neigh[IX] = 0;
          else                       // Corner on the left side;   Cases 6 and 8
            coord_neigh[IX] = bx-ghostWidth/2;

          if (corner & CORNER_TOP)   // Corner on the top side;    Cases 7 and 8
            coord_neigh[IY] = by/2;
          else                       // Corner on the bottom side; Cases 5 and 6
            coord_neigh[IY] = (by-ghostWidth)/2;

          // No need to stay here
          return;
        }
      }
      // If the corresponding neighbour is bigger, we return it
      else if (Interface_flags(iOct_local) & nc_flag_y) {
        // Y interface has a larger neighbour, that means we are in cases 2, 3, 10 or 11
        if (neigh_y.size() > 0) {
          isBoundary = false;
          neigh      = neigh_y[0];
          isGhost    = isGhost_y[0];

          if (corner & CORNER_RIGHT) // Corner on the right side;  Cases 2 and 10
            coord_neigh[IX] = bx/2;
          else                       // Corner on the left side;   Cases 3 and 11
            coord_neigh[IX] = (bx-ghostWidth)/2;

          if (corner & CORNER_TOP)   // Corner on the top side;    Cases 10 and 11
            coord_neigh[IY] = 0;
          else                       // Corner on the bottom side; Cases 2 and 3
            coord_neigh[IY] = by-ghostWidth/2;
          
          // We exit
          return;
        }
        // Shouldn't come here !
        assert(false);
      } // if Interface_flags(iOct_local) & nc_flag

      // TODO : 3d here !
    } // if neigh_v.size = 0
    
  } // get_corner_neighbours
    

  // ==============================================================
  // ==============================================================
  /**
   * This routine fills in the corners of the ghosted block in case the said corner is also a corner of
   * the domain.
   * 
   * \param[in] iOct_local id of the current octant in the active group to be filled
   * \param[in] index id of the current corner cell to be modified (between 0 and ghostWidth^2-1)
   * \param[in] corner to be filled here
   **/
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_bc_corner_2d(uint32_t iOct_local, uint32_t index, uint8_t corner) const {
    // Pre-computed offsets for shifting the coordinates in the right reference frame :
    // x/y_offset are for the neighbour block
    // gx/gy_offset are for the ghosted block
    const uint32_t gx_offset = bx+ghostWidth;
    const uint32_t gy_offset = by+ghostWidth;

    // Coordinates in the ghosted block and in the neighbour
    coord_t coord_ghost = index_to_coord(index, ghostWidth, ghostWidth);

    // Shifting ghosted index to correct position
    if (corner & 1)
      coord_ghost[IX] += gx_offset;
    if (corner & 2)
      coord_ghost[IY] += gy_offset;

    uint32_t ighost = coord_ghost[IX] + coord_ghost[IY] * bx_g;
    
    Ugroup(ighost, fm[ID], iOct_local) = 0.0;
    Ugroup(ighost, fm[IU], iOct_local) = 0.0;
    Ugroup(ighost, fm[IV], iOct_local) = 0.0;
    Ugroup(ighost, fm[IP], iOct_local) = 0.0;

    if (copy_gravity) {
      Ugroup(ighost, fm[IGX], iOct_local) = 0.0;
      Ugroup(ighost, fm[IGY], iOct_local) = 0.0;
    }
  } // fill_ghost_corner_bc_corner_2d

  // ==============================================================
  // ==============================================================
  /**
   * This routine fills in the corners of the ghosted block in case one of the edges is a boundary of 
   * the domain. This routine solely used already copied data and hence does not require any neighbour
   * information
   *
   * \param[in] iOct_local id of the current octant in the active group to be filled
   * \param[in] index id of the current corner cell to be modified (between 0 and ghostWidth^2-1)
   * \param[in] corner to be filled here
   * \param[in] dir direction aong which the boundary is
   * \param[in] face the boundary's corresponding face
   **/
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_bc_face_2d(uint32_t iOct_local, uint32_t index, uint8_t corner, uint8_t dir, uint8_t face) const {
    // Pre-computed offsets for shifting the coordinates in the right reference frame :
    // x/y_offset are for the neighbour block
    // gx/gy_offset are for the ghosted block
    const uint32_t gx_offset = bx+ghostWidth;
    const uint32_t gy_offset = by+ghostWidth;

    // Coordinates in the ghosted block and in the neighbour
    coord_t coord_ghost = index_to_coord(index, ghostWidth, ghostWidth);
    uint32_t x = coord_ghost[IX];
    uint32_t y = coord_ghost[IY];

    // Shifting ghosted index to correct position
    if (corner & CORNER_RIGHT)
      coord_ghost[IX] += gx_offset;
    if (corner & CORNER_TOP)
      coord_ghost[IY] += gy_offset;

    
    coord_t coord_copy  = coord_ghost;

    real_t sign_u = 1.0;
    real_t sign_v = 1.0;

    if (dir == DIR_X) {
      // Absorbing conditions
      if (params.boundary_type_xmin == BC_ABSORBING and face == FACE_LEFT)
	      coord_copy[IX] = ghostWidth;
      else if (params.boundary_type_xmax == BC_ABSORBING and face == FACE_RIGHT)
	      coord_copy[IX] = gx_offset-1;
      // Reflexive conditions
      else if (params.boundary_type_xmin == BC_REFLECTING and face == FACE_LEFT) {
	      coord_copy[IX] = 2*ghostWidth-1-x;
	      sign_u = -1.0;
      }
      else if (params.boundary_type_xmax == BC_REFLECTING and face == FACE_RIGHT) {
        coord_copy[IX] = bx+ghostWidth-1-x;
        sign_u = -1.0;
      }
    }
    else { // DIR_Y
      // Absorbing conditions
      if (params.boundary_type_ymin == BC_ABSORBING and face == FACE_LEFT)
	      coord_copy[IY] = ghostWidth;
      else if (params.boundary_type_ymax == BC_ABSORBING and face == FACE_RIGHT)
	      coord_copy[IY] = gy_offset-1;
      // Reflexive conditions
      else if (params.boundary_type_ymin == BC_REFLECTING and face == FACE_LEFT) {
	      coord_copy[IY] = 2*ghostWidth-1-y;
	      sign_v = -1.0;
      }
      else if (params.boundary_type_ymax == BC_REFLECTING and face == FACE_RIGHT) {
	      coord_copy[IY] = by+ghostWidth-1-y;
	      sign_v = -1.0;
      }
    } // end if dir

    uint32_t index_ghost = coord_ghost[IX] + (bx_g)*coord_ghost[IY];
    uint32_t index_copy  = coord_copy[IX]  + (bx_g)*coord_copy[IY];

    Ugroup(index_ghost, fm[ID], iOct_local) = Ugroup(index_copy, fm[ID], iOct_local);
    Ugroup(index_ghost, fm[IU], iOct_local) = Ugroup(index_copy, fm[IU], iOct_local) * sign_u;
    Ugroup(index_ghost, fm[IV], iOct_local) = Ugroup(index_copy, fm[IV], iOct_local) * sign_v;
    Ugroup(index_ghost, fm[IP], iOct_local) = Ugroup(index_copy, fm[IP], iOct_local);

    if (copy_gravity) {
      Ugroup(index_ghost, fm[IGX], iOct_local) = Ugroup(index_copy, fm[IGX], iOct_local);
      Ugroup(index_ghost, fm[IGY], iOct_local) = Ugroup(index_copy, fm[IGY], iOct_local);
    }
  } // fill_ghost_corner_bc_face_2d

  // ==============================================================
  // ==============================================================
  /**
   * This routine fills in the corners of a ghosted block if the corresponding neighbour is one level
   * coarser than the current block.
   *
   * \param[in] iOct_local id of the current octant in the active group to be filled
   * \param[in] index id of the current corner cell to be modified (between 0 and ghostWidth^2-1)
   * \param[in] corner to be filled here
   * \param[in] iOct_neigh id of the neighbouring octant corresponding to the corner
   * \param[in] isGhost indicates if the neighbour has to be copied from the U_ghost
   * \param[in] coord_neigh the coordinates in the neighbour, that have already been returned by get_corner_neighbours
   **/
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_larger_2d(uint32_t iOct_local, uint32_t iOct, uint32_t index, uint8_t corner, uint32_t iOct_neigh,
				   bool isGhost, coord_t base_neigh) const {
    // Pre-computed offsets for shifting coordinates in the right reference frame
    // gx/gy_offset are for the ghosted block
    uint32_t gx_offset = bx+ghostWidth;
    uint32_t gy_offset = by+ghostWidth;

    // Cordinates in the neighbour and in the ghosted block
    coord_t coord_ghost = index_to_coord(index, ghostWidth, ghostWidth);
    coord_t coord_neigh = base_neigh;
    
    // We offset the neighbour coords by the current position in the corner block
    // WARNING ! Check that this is correct with bx % 2 = 1 !!!
    coord_neigh[IX] += coord_ghost[IX] / 2;
    coord_neigh[IY] += coord_ghost[IY] / 2;

    // Shifting the positions to the correct ids
    if (corner & CORNER_RIGHT) // Right side
      coord_ghost[IX] += gx_offset;
    if (corner & CORNER_TOP)   // Top side
      coord_ghost[IY] += gy_offset;

    uint32_t ghost_index = coord_ghost[IX] + bx_g*coord_ghost[IY];
    uint32_t neigh_index = coord_neigh[IX] + bx*coord_neigh[IY];

    if (isGhost) {
      Ugroup(ghost_index, fm[ID], iOct_local) = U_ghost(neigh_index, fm[ID], iOct_neigh);
      Ugroup(ghost_index, fm[IU], iOct_local) = U_ghost(neigh_index, fm[IU], iOct_neigh);
      Ugroup(ghost_index, fm[IV], iOct_local) = U_ghost(neigh_index, fm[IV], iOct_neigh);
      Ugroup(ghost_index, fm[IP], iOct_local) = U_ghost(neigh_index, fm[IP], iOct_neigh);

      if (copy_gravity) {
        Ugroup(ghost_index, fm[IGX], iOct_local) = U_ghost(neigh_index, fm[IGX], iOct_neigh);
        Ugroup(ghost_index, fm[IGY], iOct_local) = U_ghost(neigh_index, fm[IGY], iOct_neigh);
      }
    }
    else {
      Ugroup(ghost_index, fm[ID], iOct_local) = U(neigh_index, fm[ID], iOct_neigh);
      Ugroup(ghost_index, fm[IU], iOct_local) = U(neigh_index, fm[IU], iOct_neigh);
      Ugroup(ghost_index, fm[IV], iOct_local) = U(neigh_index, fm[IV], iOct_neigh);
      Ugroup(ghost_index, fm[IP], iOct_local) = U(neigh_index, fm[IP], iOct_neigh);

      if (copy_gravity) {
        Ugroup(ghost_index, fm[IGX], iOct_local) = U(neigh_index, fm[IGX], iOct_neigh);
        Ugroup(ghost_index, fm[IGY], iOct_local) = U(neigh_index, fm[IGY], iOct_neigh);
      }
    } // end ifGhost
  } // fill_ghost_corner_larger_2d

  // ==============================================================
  // ==============================================================
  /**
   * This routine fills in the corners of a ghosted block if the corresponding neighbour is one level
   * finer than the current block.
   *
   * \param[in] iOct_local id of the current octant in the active group to be filled
   * \param[in] index id of the current corner cell to be modified (between 0 and ghostWidth^2-1)
   * \param[in] corner to be filled here
   * \param[in] iOct_neigh id of the neighbouring octant corresponding to the corner
   * \param[in] isGhost indicates if the neighbour has to be copied from the U_ghost
   **/
  KOKKOS_INLINE_FUNCTION
    void fill_ghost_corner_smaller_2d(uint32_t iOct_local, uint32_t iOct, uint32_t index, uint8_t corner, uint32_t iOct_neigh,
				    bool isGhost) const {
    // Pre-computed offsets for shifting coordinates in the right reference frame
    // x/y_offsets are for the neighbour cell
    // gx/gy_offsets aree for the ghosted block
    const uint32_t x_offset  = bx-ghostWidth*2;
    const uint32_t y_offset  = by-ghostWidth*2;
    const uint32_t gx_offset = bx+ghostWidth;
    const uint32_t gy_offset = by+ghostWidth;

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
    else                       // Bottom side
      coord_neigh[IY] += y_offset;

    // We accumulate the results in this variable
    HydroState2d u = {0.0, 0.0, 0.0, 0.0};
    real_t gravity[2] {0.0, 0.0};

    uint32_t ghost_index = coord_ghost[IX] + bx_g*coord_ghost[IY];

    // Summing all the sub-cells
    for (int ix=0; ix < 2; ++ix) {
      for (int iy=0; iy < 2; ++iy) {
        coord_t coord_sub = coord_neigh;
        coord_sub[IX] += ix;
        coord_sub[IY] += iy;

        uint32_t neigh_index = coord_sub[IX] + bx*coord_sub[IY];

        if (isGhost) {
          u[ID] += U_ghost(neigh_index, fm[ID], iOct_neigh);
          u[IU] += U_ghost(neigh_index, fm[IU], iOct_neigh);
          u[IV] += U_ghost(neigh_index, fm[IV], iOct_neigh);
          u[IP] += U_ghost(neigh_index, fm[IP], iOct_neigh);

          if (copy_gravity) {
            gravity[0] += U_ghost(neigh_index, fm[IGX], iOct_neigh);
            gravity[1] += U_ghost(neigh_index, fm[IGY], iOct_neigh);
          }
        }
        else {
          u[ID] += U(neigh_index, fm[ID], iOct_neigh);
          u[IU] += U(neigh_index, fm[IU], iOct_neigh);
          u[IV] += U(neigh_index, fm[IV], iOct_neigh);
          u[IP] += U(neigh_index, fm[IP], iOct_neigh);

          if (copy_gravity) {
            gravity[0] += U(neigh_index, fm[IGX], iOct_neigh);
            gravity[1] += U(neigh_index, fm[IGY], iOct_neigh);
          }
        } // end isGhost
      } // end for iy
    } // end for ix

    // And averaging
    Ugroup(ghost_index, fm[ID], iOct_local) = 0.25 * u[ID];
    Ugroup(ghost_index, fm[IU], iOct_local) = 0.25 * u[IU];
    Ugroup(ghost_index, fm[IV], iOct_local) = 0.25 * u[IV];
    Ugroup(ghost_index, fm[IP], iOct_local) = 0.25 * u[IP];

    if (copy_gravity) { 
      Ugroup(ghost_index, fm[IGX], iOct_local) = 0.25 * gravity[0];
      Ugroup(ghost_index, fm[IGY], iOct_local) = 0.25 * gravity[1];
    }

  } // fill_ghost_corner_smaller_2d

  // ==============================================================
  // ==============================================================
  /**
   * This routine fills in the corners of a ghosted block if the corresponding neighbour is on the same
   * level.
   *
   * \param[in] iOct_local id of the current octant in the active group to be filled
   * \param[in] index id of the current corner cell to be modified (between 0 and ghostWidth^2-1)
   * \param[in] corner to be filled here
   * \param[in] iOct_neigh id of the neighbouring octant corresponding to the corner
   * \param[in] isGhost indicates if the neighbour has to be copied from the U_ghost
   **/
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_same_size_2d(uint32_t iOct_local, uint32_t iOct, uint32_t index, uint8_t corner, uint32_t iOct_neigh,
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

    // And we copy the data
    if (isGhost) {
      Ugroup(ghost_index, fm[ID], iOct_local) = U_ghost(neigh_index, fm[ID], iOct_neigh);
      Ugroup(ghost_index, fm[IU], iOct_local) = U_ghost(neigh_index, fm[IU], iOct_neigh);
      Ugroup(ghost_index, fm[IV], iOct_local) = U_ghost(neigh_index, fm[IV], iOct_neigh);
      Ugroup(ghost_index, fm[IP], iOct_local) = U_ghost(neigh_index, fm[IP], iOct_neigh);

      if (copy_gravity) {
        Ugroup(ghost_index, fm[IGX], iOct_local) = U_ghost(neigh_index, fm[IGX], iOct_neigh);
        Ugroup(ghost_index, fm[IGY], iOct_local) = U_ghost(neigh_index, fm[IGY], iOct_neigh);
      }
    }
    else {
      Ugroup(ghost_index, fm[ID], iOct_local) = U(neigh_index, fm[ID], iOct_neigh);
      Ugroup(ghost_index, fm[IU], iOct_local) = U(neigh_index, fm[IU], iOct_neigh);
      Ugroup(ghost_index, fm[IV], iOct_local) = U(neigh_index, fm[IV], iOct_neigh);
      Ugroup(ghost_index, fm[IP], iOct_local) = U(neigh_index, fm[IP], iOct_neigh);

      if (copy_gravity) {
        Ugroup(ghost_index, fm[IGX], iOct_local) = U(neigh_index, fm[IGX], iOct_neigh);
        Ugroup(ghost_index, fm[IGY], iOct_local) = U(neigh_index, fm[IGY], iOct_neigh);
      }
    }

  } // fill_ghost_corner_same_size_2d

  // ==============================================================
  // ==============================================================
  /** 
   * This routine is a proxy that finds in which of the following routines has to be called :
   *  . fill_ghost_corner_same_size_2d
   *  . fill_ghost_corner_larger_2d
   *  . fill_ghost_corner_smaller_2d
   *  . fill_ghost_corner_bc_face_2d
   *  . fill_ghost_corner_bc_corner_2d   
   *
   * \param[in] iOct_local id of the current octant in the active group to be filled
   * \param[in] index id of the current corner cell to be modified (between 0 and ghostWidth^2-1)
   * \param[in] corner to be filled here
   **/
  KOKKOS_INLINE_FUNCTION
  void fill_ghost_corner_2d(uint32_t iOct, uint32_t iOct_local, uint32_t index, uint8_t corner) const {
    uint32_t neigh;
    bool isGhost, isBoundary;
    coord_t neigh_coords;

    // We retrieve the corner ids, wether we're hitting a boundary, and if the location is shifted
    neigh=0;
    isGhost=false;
    get_corner_neighbours(iOct, iOct_local, corner, isBoundary, neigh, isGhost, neigh_coords);
    
    // We treat boundary conditions differently
    if (isBoundary) {
      // We check if we have one or two boundaries
      const uint8_t face_codim = 1;
      std::vector<uint32_t> n1, n2;
      std::vector<bool> ig1, ig2;

      // Extracting the two faces of the corner
      // iface1 is along the X direction
      // iface2 along the Y direction
      const uint8_t iface1 = (corner & 1 ? FACE_XMAX : FACE_XMIN);
      const uint8_t iface2 = (corner & 2 ? FACE_YMAX : FACE_YMIN);

      // We extract the neighbours to find if the corner is crossing one or two boundaries
      pmesh->findNeighbours(iOct, iface1, face_codim, n1, ig1);
      pmesh->findNeighbours(iOct, iface2, face_codim, n2, ig2);

      // Faces of copy
      const uint8_t f1 = (corner & 1 ? FACE_RIGHT : FACE_LEFT);
      const uint8_t f2 = (corner & 2 ? FACE_RIGHT : FACE_LEFT);
      
      if (n2.size() > 0)      // X face is boundary
	      fill_ghost_corner_bc_face_2d(iOct_local, index, corner, DIR_X, f1);
      else if (n1.size() > 0) // Y face is boundary
	      fill_ghost_corner_bc_face_2d(iOct_local, index, corner, DIR_Y, f2);
      else                    // X and Y are boundaries
	      fill_ghost_corner_bc_corner_2d(iOct_local, index, corner);
    }
    else {  // If we have a neighbour, we treat the different AMR cases separately
      uint32_t cur_level, neigh_level;

      cur_level   = pmesh->getLevel(iOct);
      if (isGhost)
        neigh_level = pmesh->getLevel(pmesh->getGhostOctant(neigh));
      else
        neigh_level = pmesh->getLevel(neigh);

      if (cur_level == neigh_level)     // Same level
	      fill_ghost_corner_same_size_2d(iOct_local, iOct, index, corner, neigh, isGhost);
      else if (cur_level > neigh_level) // Larger neighbour
	      fill_ghost_corner_larger_2d(iOct_local, iOct, index, corner, neigh, isGhost, neigh_coords);
      else                              // Smaller neighbour
	      fill_ghost_corner_smaller_2d(iOct_local, iOct, index, corner, neigh, isGhost);
    } // end if neigh.size() == 
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

  // ! flag array to keep track of which side is non-conformal 
  FlagArrayBlock Interface_flags;

  // ! epsilon value to test equality between reals
  real_t eps;

  // should we copy gravity ?
  bool copy_gravity;

  // number of dimensions for gravity
  int ndim; 

}; // CopyCornerBlockCellDataFunctor

} // namespace muscl_block
} // namespace dyablo

#endif
